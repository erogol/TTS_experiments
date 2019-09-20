import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from TTS.datasets.TTSDataset import MyDataset
from distribute import (DistributedSampler, apply_gradient_allreduce,
                        init_distributed, reduce_tensor)
from TTS.layers.losses import L1LossMasked, MSELossMasked
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import (NoamLR, check_update, count_parameters,
                                     create_experiment_folder, get_git_branch,
                                     load_config, remove_experiment_folder,
                                     save_best_model, save_checkpoint, weight_decay,
                                     set_init_dict, copy_config_file, setup_model,
                                     split_dataset, gradual_training_scheduler)
from TTS.utils.logger import Logger
from TTS.utils.speakers import load_speaker_mapping, save_speaker_mapping, \
    get_speakers
from TTS.utils.synthesis import synthesis
from TTS.utils.text.symbols import phonemes, symbols
from TTS.utils.visual import plot_alignment, plot_spectrogram
from TTS.datasets.preprocess import get_preprocessor_by_name
from TTS.utils.radam import RAdam
from TTS.utils.measures import alignment_diagonal_score
from TTS.utils.generic_utils import sequence_mask

## Duration predictor
from TTS.models.duration_predictor import DurationPredictor, DurationLoss

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(" > Using CUDA: ", use_cuda)
print(" > Number of GPUs: ", num_gpus)


def setup_loader(ap, is_val=False, verbose=False):
    global meta_data_train
    global meta_data_eval
    if "meta_data_train" not in globals():
        if c.meta_file_train is not None:
            meta_data_train = get_preprocessor_by_name(c.dataset)(c.data_path, c.meta_file_train)
        else:
            meta_data_train = get_preprocessor_by_name(c.dataset)(c.data_path)
    if "meta_data_eval" not in globals() and c.run_eval:
        if c.meta_file_val is not None:
            meta_data_eval = get_preprocessor_by_name(c.dataset)(c.data_path, c.meta_file_val)
        else:
            meta_data_eval, meta_data_train = split_dataset(meta_data_train)
    if is_val and not c.run_eval:
        loader = None
    else:
        dataset = MyDataset(
            c.r,
            c.text_cleaner,
            meta_data=meta_data_eval if is_val else meta_data_train,
            ap=ap,
            batch_group_size=0 if is_val else c.batch_group_size * c.batch_size,
            min_seq_len=c.min_seq_len,
            max_seq_len=c.max_seq_len,
            phoneme_cache_path=c.phoneme_cache_path,
            use_phonemes=c.use_phonemes,
            phoneme_language=c.phoneme_language,
            enable_eos_bos=c.enable_eos_bos_chars,
            verbose=verbose)
        sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=c.eval_batch_size if is_val else c.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            sampler=sampler,
            num_workers=c.num_val_loader_workers
            if is_val else c.num_loader_workers,
            pin_memory=False)
    return loader


def train(model, criterion, criterion_st, optimizer, optimizer_st, scheduler,
          ap, global_step, epoch, model_duration, criterion_duration, optimizer_duration):
    data_loader = setup_loader(ap, is_val=False, verbose=(epoch == 0))
    if c.use_speaker_embedding:
        speaker_mapping = load_speaker_mapping(OUT_PATH)
    model.train()
    epoch_time = 0
    avg_postnet_loss = 0
    avg_decoder_loss = 0
    avg_alignment_score = 0
    avg_stop_loss = 0
    avg_step_time = 0
    avg_loader_time = 0
    avg_duration_loss = 0
    print("\n > Epoch {}/{}".format(epoch, c.epochs), flush=True)
    if use_cuda:
        batch_n_iter = int(len(data_loader.dataset) / (c.batch_size * num_gpus))
    else:
        batch_n_iter = int(len(data_loader.dataset) / c.batch_size)
    end_time = time.time()
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()

        # setup input data
        text_input = data[0]
        text_lengths = data[1]
        speaker_names = data[2]
        linear_input = data[3] if c.model in ["Tacotron", "TacotronGST"] else None
        mel_input = data[4]
        mel_lengths = data[5]
        stop_targets = data[6]
        avg_text_length = torch.mean(text_lengths.float())
        avg_spec_length = torch.mean(mel_lengths.float())
        loader_time = time.time() - end_time

        if c.use_speaker_embedding:
            speaker_ids = [speaker_mapping[speaker_name]
                           for speaker_name in speaker_names]
            speaker_ids = torch.LongTensor(speaker_ids)
        else:
            speaker_ids = None

        # set stop targets view, we predict a single stop token per r frames prediction
        stop_targets = stop_targets.view(text_input.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze(2)

        global_step += 1

        # setup lr
        if c.lr_decay:
            scheduler.step()
        optimizer.zero_grad()
        if optimizer_st:
            optimizer_st.zero_grad()

        # dispatch data to GPU
        if use_cuda:
            text_input = text_input.cuda(non_blocking=True)
            text_lengths = text_lengths.cuda(non_blocking=True)
            mel_input = mel_input.cuda(non_blocking=True)
            mel_lengths = mel_lengths.cuda(non_blocking=True)
            linear_input = linear_input.cuda(non_blocking=True) if c.model in ["Tacotron", "TacotronGST"] else None
            stop_targets = stop_targets.cuda(non_blocking=True)
            if speaker_ids is not None:
                speaker_ids = speaker_ids.cuda(non_blocking=True)

        # forward pass model
        decoder_output, postnet_output, alignments, scale_factor = model.forward_duration(
            text_input, text_lengths, mel_input, model_duration, speaker_ids=speaker_ids)

        # loss computation
        mel_input = torch.nn.functional.interpolate(mel_input.transpose(1, 2), size=scale_factor).transpose(1, 2)
        linear_input = torch.nn.functional.interpolate(linear_input.transpose(1, 2), size=scale_factor).transpose(1, 2)
        if c.loss_masking:
            decoder_loss = criterion(decoder_output, mel_input, mel_lengths)
            if c.model in ["Tacotron", "TacotronGST"]:
                postnet_loss = criterion(postnet_output, linear_input, mel_lengths)
            else:
                postnet_loss = criterion(postnet_output, mel_input, mel_lengths)
        else:
            decoder_loss = criterion(decoder_output, mel_input)
            if c.model in ["Tacotron", "TacotronGST"]:
                postnet_loss = criterion(postnet_output, linear_input)
            else:
                postnet_loss = criterion(postnet_output, mel_input)
        loss = decoder_loss + postnet_loss
        # if not c.separate_stopnet and c.stopnet:
        #     loss += stop_loss

         ## DURATION MODEL
        # optimizer_duration.zero_grad()
        # with torch.no_grad():
        #     durations = model_duration.calculate_durations(alignments.(), text_lengths.detach().cpu().numpy(), mel_lengths.detach().cpu().numpy())
        #     scores = model_duration.calculate_scores(alignments.detach(), text_lengths.detach().cpu().numpy(), mel_lengths.detach().cpu().numpy())
        # duration_pred, scores_pred = model_duration.forward(model.encoder_outputs,)
        # duration_loss = criterion_duration(duration_pred, durations, scores_pred, scores)
        # duration_loss.backward()
        # optimizer_duration.step()
        ## END DURATION MODEL

        loss.backward()
        optimizer, current_lr = weight_decay(optimizer, c.wd)
        grad_norm, _ = check_update(model, c.grad_clip)
        optimizer.step()

        # compute alignment score
        alignment_score = alignment_diagonal_score(alignments)
        avg_alignment_score += alignment_score

        # backpass and check the grad norm for stop loss
        if c.separate_stopnet:
            # stop_loss.backward()
            optimizer_st, _ = weight_decay(optimizer_st, c.wd)
            grad_norm_st, _ = check_update(model.decoder.stopnet, 1.0)
            optimizer_st.step()
        else:
            grad_norm_st = 0

        step_time = time.time() - start_time
        epoch_time += step_time

        if global_step % c.print_step == 0:
            print(
                "   | > Step:{}/{}  GlobalStep:{}  TotalLoss:{:.5f}  PostnetLoss:{:.5f}  "
                "DecoderLoss:{:.5f}  StopLoss:{:.5f}  DurLoss:{:.5f}  AlignScore:{:.4}  GradNorm:{:.5f}  "
                "GradNormST:{:.5f}  AvgTextLen:{:.1f}  AvgSpecLen:{:.1f}  StepTime:{:.2f}  "
                "LoaderTime:{:.2f}  LR:{:.6f}".format(
                    num_iter, batch_n_iter, global_step, loss.item(),
                    postnet_loss.item(), decoder_loss.item(), 0, 0, alignment_score.item(),
                    grad_norm, grad_norm_st, avg_text_length, avg_spec_length, step_time,
                    loader_time, current_lr),
                flush=True)

        # aggregate losses from processes
        if num_gpus > 1:
            postnet_loss = reduce_tensor(postnet_loss.data, num_gpus)
            decoder_loss = reduce_tensor(decoder_loss.data, num_gpus)
            loss = reduce_tensor(loss.data, num_gpus)
            # stop_loss = reduce_tensor(stop_loss.data, num_gpus) if c.stopnet else stop_loss

        if args.rank == 0:
            avg_postnet_loss += float(postnet_loss.item())
            avg_decoder_loss += float(decoder_loss.item())
            # avg_duration_loss += float(duration_loss.item())
            # avg_stop_loss += stop_loss if isinstance(stop_loss, float) else float(stop_loss.item())
            avg_step_time += step_time
            avg_loader_time += loader_time

            # Plot Training Iter Stats
            # reduce TB load
            if global_step % 10 == 0:
                iter_stats = {"loss_posnet": postnet_loss.item(),
                              "loss_decoder": decoder_loss.item(),
                              "lr": current_lr,
                              "grad_norm": grad_norm,
                              "grad_norm_st": grad_norm_st,
                              "step_time": step_time}
                tb_logger.tb_train_iter_stats(global_step, iter_stats)

            if global_step % c.save_step == 0:
                if c.checkpoint:
                    # save model
                    save_checkpoint(model, model_duration, optimizer, optimizer_st,
                                    postnet_loss.item(), OUT_PATH, global_step,
                                    epoch)

                # Diagnostic visualizations
                const_spec = postnet_output[0].data.cpu().numpy()
                gt_spec = linear_input[0].data.cpu().numpy() if c.model in ["Tacotron", "TacotronGST"] else  mel_input[0].data.cpu().numpy()
                align_img = alignments[0].data.cpu().numpy()

                figures = {
                    "prediction": plot_spectrogram(const_spec, ap),
                    "ground_truth": plot_spectrogram(gt_spec, ap),
                    "alignment": plot_alignment(align_img)
                }
                tb_logger.tb_train_figures(global_step, figures)

                # Sample audio
                if c.model in ["Tacotron", "TacotronGST"]:
                    train_audio = ap.inv_spectrogram(const_spec.T)
                else:
                    train_audio = ap.inv_mel_spectrogram(const_spec.T)
                tb_logger.tb_train_audios(global_step,
                                          {'TrainAudio': train_audio},
                                          c.audio["sample_rate"])
        end_time = time.time()

    avg_postnet_loss /= (num_iter + 1)
    avg_decoder_loss /= (num_iter + 1)
    # avg_stop_loss /= (num_iter + 1)
    # avg_duration_loss /= (num_iter + 1)
    avg_total_loss = avg_decoder_loss + avg_postnet_loss #+ avg_stop_loss
    avg_step_time /= (num_iter + 1)
    avg_loader_time /= (num_iter + 1)
    avg_alignment_score /= (num_iter + 1)

    # print epoch stats
    print(
        "   | > EPOCH END -- GlobalStep:{}  AvgTotalLoss:{:.5f}  "
        "AvgPostnetLoss:{:.5f}  AvgDecoderLoss:{:.5f}  AvgDurationLoss:{:.5f}  AvgAlignScore:{:.3f}  "
        "AvgStopLoss:{:.5f}  EpochTime:{:.2f}  "
        "AvgStepTime:{:.2f}  AvgLoaderTime:{:.2f}".format(global_step, avg_total_loss,
                                                          avg_postnet_loss, avg_decoder_loss, 
                                                          avg_duration_loss, avg_alignment_score,
                                                          0, epoch_time, avg_step_time,
                                                          avg_loader_time),
        flush=True)

    # Plot Epoch Stats
    if args.rank == 0:
        # Plot Training Epoch Stats
        epoch_stats = {"loss_postnet": avg_postnet_loss,
                       "loss_decoder": avg_decoder_loss,
                    #    "stop_loss": avg_stop_loss,
                       "alignment_score": avg_alignment_score,
                    #    "duration_loss": avg_duration_loss,
                       "epoch_time": epoch_time}
        tb_logger.tb_train_epoch_stats(global_step, epoch_stats)
        if c.tb_model_param_stats:
            tb_logger.tb_model_weights(model, global_step)
    return avg_postnet_loss, global_step


def evaluate(model, criterion, criterion_st, ap, global_step, epoch, model_duration, criterion_duration):
    data_loader = setup_loader(ap, is_val=True)
    if c.use_speaker_embedding:
        speaker_mapping = load_speaker_mapping(OUT_PATH)
    model.eval()
    epoch_time = 0
    avg_postnet_loss = 0
    avg_decoder_loss = 0
    avg_stop_loss = 0
    avg_duration_loss = 0
    print("\n > Validation")
    if c.test_sentences_file is None:
        test_sentences = [
            "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
            "Be a voice, not an echo.",
            "I'm sorry Dave. I'm afraid I can't do that.",
            "This cake is great. It's so delicious and moist."
        ]
    else:
        with open(c.test_sentences_file, "r") as f:
            test_sentences = [s.strip() for s in f.readlines()]
    with torch.no_grad():
        if data_loader is not None:
            for num_iter, data in enumerate(data_loader):
                start_time = time.time()

                # setup input data
                text_input = data[0]
                text_lengths = data[1]
                speaker_names = data[2]
                linear_input = data[3] if c.model in ["Tacotron", "TacotronGST"] else None
                mel_input = data[4]
                mel_lengths = data[5]
                stop_targets = data[6]

                if c.use_speaker_embedding:
                    speaker_ids = [speaker_mapping[speaker_name]
                                   for speaker_name in speaker_names]
                    speaker_ids = torch.LongTensor(speaker_ids)
                else:
                    speaker_ids = None

                # set stop targets view, we predict a single stop token per r frames prediction
                stop_targets = stop_targets.view(text_input.shape[0],
                                                 stop_targets.size(1) // c.r,
                                                 -1)
                stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze(2)

                # dispatch data to GPU
                if use_cuda:
                    text_input = text_input.cuda()
                    mel_input = mel_input.cuda()
                    mel_lengths = mel_lengths.cuda()
                    linear_input = linear_input.cuda() if c.model in ["Tacotron", "TacotronGST"] else None
                    stop_targets = stop_targets.cuda()
                    if speaker_ids is not None:
                        speaker_ids = speaker_ids.cuda()

                # forward pass
                decoder_output, postnet_output, alignments, scale_factor =\
                    model.forward_duration(text_input, text_lengths, mel_input, model_duration,
                                  speaker_ids=speaker_ids)

                # loss computation
                mel_input = torch.nn.functional.interpolate(mel_input.transpose(1, 2), size=scale_factor).transpose(1, 2)
                linear_input = torch.nn.functional.interpolate(linear_input.transpose(1, 2), size=scale_factor).transpose(1, 2) 
                if c.loss_masking:              
                    decoder_loss = criterion(decoder_output, mel_input, mel_lengths)
                    if c.model in ["Tacotron", "TacotronGST"]:
                        postnet_loss = criterion(postnet_output, linear_input, mel_lengths)
                    else:
                        postnet_loss = criterion(postnet_output, mel_input, mel_lengths)
                else:
                    decoder_loss = criterion(decoder_output, mel_input)
                    if c.model in ["Tacotron", "TacotronGST"]:
                        postnet_loss = criterion(postnet_output, linear_input)
                    else:
                        postnet_loss = criterion(postnet_output, mel_input)
                loss = decoder_loss + postnet_loss # + stop_loss

                ## DURATION MODEL
                # durations = model_duration.calculate_durations(alignments.detach(), text_lengths.detach().cpu().numpy(), mel_lengths.detach().cpu().numpy())
                # scores = model_duration.calculate_scores(alignments.detach(), text_lengths.detach().cpu().numpy(), mel_lengths.detach().cpu().numpy())
                # duration_pred, scores_pred = model_duration.forward(model.encoder_outputs.detach())
                # duration_loss = criterion_duration(duration_pred, durations, scores_pred, scores)
                ## END DURATION MODEL

                step_time = time.time() - start_time
                epoch_time += step_time

                if num_iter % c.print_step == 0:
                    print(
                        "   | > TotalLoss: {:.5f}   PostnetLoss: {:.5f}   DecoderLoss: {:.5f}  DurationLoss: {:.5f}  "
                        "StopLoss: {:.5f}  ".format(loss.item(),
                                                    postnet_loss.item(),
                                                    decoder_loss.item(),
                                                    0,
                                                    0),
                        flush=True)

                # aggregate losses from processes
                if num_gpus > 1:
                    postnet_loss = reduce_tensor(postnet_loss.data, num_gpus)
                    decoder_loss = reduce_tensor(decoder_loss.data, num_gpus)
                    # if c.stopnet:
                        # stop_loss = reduce_tensor(stop_loss.data, num_gpus)

                avg_postnet_loss += float(postnet_loss.item())
                avg_decoder_loss += float(decoder_loss.item())
                # avg_duration_loss += float(duration_loss.item())
                # avg_stop_loss += stop_loss.item()

            if args.rank == 0:
                # Diagnostic visualizations
                idx = np.random.randint(mel_input.shape[0])
                const_spec = postnet_output[idx].data.cpu().numpy()
                gt_spec = linear_input[idx].data.cpu().numpy() if c.model in ["Tacotron", "TacotronGST"] else  mel_input[idx].data.cpu().numpy()
                align_img = alignments[idx].data.cpu().numpy()

                eval_figures = {
                    "prediction": plot_spectrogram(const_spec, ap),
                    "ground_truth": plot_spectrogram(gt_spec, ap),
                    "alignment": plot_alignment(align_img)
                }
                tb_logger.tb_eval_figures(global_step, eval_figures)

                # Sample audio
                if c.model in ["Tacotron", "TacotronGST"]:
                    eval_audio = ap.inv_spectrogram(const_spec.T)
                else:
                    eval_audio = ap.inv_mel_spectrogram(const_spec.T)
                tb_logger.tb_eval_audios(global_step, {"ValAudio": eval_audio}, c.audio["sample_rate"])

                # compute average losses
                avg_postnet_loss /= (num_iter + 1)
                avg_decoder_loss /= (num_iter + 1)
                # avg_stop_loss /= (num_iter + 1)

                # Plot Validation Stats
                epoch_stats = {"loss_postnet": avg_postnet_loss,
                               "loss_decoder": avg_decoder_loss,}
                            #    "loss_duration": avg_duration_loss}
                            #    "stop_loss": avg_stop_loss}
                tb_logger.tb_eval_stats(global_step, epoch_stats)

    if args.rank == 0 and epoch > c.test_delay_epochs:
        # test sentences
        test_audios = {}
        test_figures = {}
        print(" | > Synthesizing test sentences")
        speaker_id = 0 if c.use_speaker_embedding else None
        style_wav = c.get("style_wav_for_test")
        for idx, test_sentence in enumerate(test_sentences):
            try:
                wav, alignment, decoder_output, postnet_output, _ = synthesis(
                    model, test_sentence, c, use_cuda, ap,
                    speaker_id=speaker_id,
                    style_wav=style_wav)
                file_path = os.path.join(AUDIO_PATH, str(global_step))
                os.makedirs(file_path, exist_ok=True)
                file_path = os.path.join(file_path,
                                         "TestSentence_{}.wav".format(idx))
                ap.save_wav(wav, file_path)
                test_audios['{}-audio'.format(idx)] = wav
                test_figures['{}-prediction'.format(idx)] = plot_spectrogram(postnet_output, ap)
                test_figures['{}-alignment'.format(idx)] = plot_alignment(alignment)
            except:
                print(" !! Error creating Test Sentence -", idx)
                traceback.print_exc()
        tb_logger.tb_test_audios(global_step, test_audios, c.audio['sample_rate'])
        tb_logger.tb_test_figures(global_step, test_figures)
    return avg_postnet_loss


#FIXME: move args definition/parsing inside of main?
def main(args): #pylint: disable=redefined-outer-name
    # Audio processor
    ap = AudioProcessor(**c.audio)

    # DISTRUBUTED
    if num_gpus > 1:
        init_distributed(args.rank, num_gpus, args.group_id,
                         c.distributed["backend"], c.distributed["url"])
    num_chars = len(phonemes) if c.use_phonemes else len(symbols)

    if c.use_speaker_embedding:
        speakers = get_speakers(c.data_path, c.meta_file_train, c.dataset)
        if args.restore_path:
            prev_out_path = os.path.dirname(args.restore_path)
            speaker_mapping = load_speaker_mapping(prev_out_path)
            assert all([speaker in speaker_mapping
                        for speaker in speakers]), "As of now you, you cannot " \
                                                   "introduce new speakers to " \
                                                   "a previously trained model."
        else:
            speaker_mapping = {name: i
                               for i, name in enumerate(speakers)}
        save_speaker_mapping(OUT_PATH, speaker_mapping)
        num_speakers = len(speaker_mapping)
        print("Training with {} speakers: {}".format(num_speakers,
                                                     ", ".join(speakers)))
    else:
        num_speakers = 0

    model = setup_model(num_chars, num_speakers, c)

    # duration model
    model_duration = DurationPredictor(256)
    criterion_duration = DurationLoss()
    optimizer_duration = None

    print(" | > Num output units : {}".format(ap.num_freq), flush=True)

    optimizer = RAdam(list(model.decoder.parameters()) + list(model.postnet.parameters()) + list(model.last_linear.parameters()), lr=c.lr, weight_decay=0)
    if c.stopnet and c.separate_stopnet:
        optimizer_st = RAdam(
            model.decoder.stopnet.parameters(), lr=c.lr, weight_decay=0)
    else:
        optimizer_st = None

    if c.loss_masking:
        criterion = L1LossMasked() if c.model in ["Tacotron", "TacotronGST"] else MSELossMasked()
    else:
        criterion = nn.L1Loss() if c.model in ["Tacotron", "TacotronGST"] else nn.MSELoss()
    criterion_st = nn.BCEWithLogitsLoss() if c.stopnet else None

    if args.restore_path:
        if num_gpus==0:
            checkpoint = torch.load(args.restore_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(args.restore_path)
        try:
            # TODO: fix optimizer init, model.cuda() needs to be called before
            # optimizer restore
            # optimizer.load_state_dict(checkpoint['optimizer'])
            if c.reinit_layers:
                raise RuntimeError
            model.load_state_dict(checkpoint['model'])
        except:
            print(" > Partial model initialization.")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint, c)
            model.load_state_dict(model_dict)
            del model_dict
        if 'model_duration' in checkpoint.keys():
            print(" > Duration model is restored.")
#             model_duration.load_state_dict(checkpoint['model_duration'])
        for group in optimizer.param_groups:
            group['lr'] = c.lr
        print(
            " > Model restored from step %d" % checkpoint['step'], flush=True)
        args.restore_step = checkpoint['step']
        model_duration.load_state_dict(checkpoint['model_duration'])
    else:
        args.restore_step = 0

    if use_cuda:
        model = model.cuda()
        criterion.cuda()
        model_duration = model_duration.cuda()
        if criterion_st:
            criterion_st.cuda()

    # DISTRUBUTED
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
        model_duration = apply_gradient_allreduce(model_duration)

    if c.lr_decay:
        scheduler = NoamLR(
            optimizer,
            warmup_steps=c.warmup_steps,
            last_epoch=args.restore_step - 1)
    else:
        scheduler = None

    num_params = count_parameters(model)
    print("\n > Model has {} parameters".format(num_params), flush=True)

    if 'best_loss' not in locals():
        best_loss = float('inf')

    global_step = args.restore_step
    for epoch in range(0, c.epochs):
        # set gradual training
        if c.gradual_training is not None:
            r, c.batch_size = gradual_training_scheduler(global_step, c)
            c.r = r
            model.decoder.set_r(r)
        print(" > Number of outputs per iteration:", model.decoder.r)

        train_loss, global_step = train(model, criterion, criterion_st,
                                        optimizer, optimizer_st, scheduler,
                                        ap, global_step, epoch, model_duration, criterion_duration, optimizer_duration)
        val_loss = evaluate(model, criterion, criterion_st, ap, global_step, epoch, model_duration, criterion_duration)
        print(
            " | > Training Loss: {:.5f}   Validation Loss: {:.5f}".format(
                train_loss, val_loss),
            flush=True)
        target_loss = train_loss
        if c.run_eval:
            target_loss = val_loss
        best_loss = save_best_model(model, model_duration, optimizer, target_loss, best_loss,
                                    OUT_PATH, global_step, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Path to model outputs (checkpoint, tensorboard etc.).',
        default=0)
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to config file for training.',
    )
    parser.add_argument(
        '--debug',
        type=bool,
        default=True,
        help='Do not verify commit integrity to run training.')
    parser.add_argument(
        '--data_path',
        type=str,
        default='',
        help='Defines the data path. It overwrites config.json.')
    parser.add_argument(
        '--output_path',
        type=str,
        help='path for training outputs.',
        default='')
    parser.add_argument(
        '--output_folder',
        type=str,
        default='',
        help='folder name for training outputs.'
    )

    # DISTRUBUTED
    parser.add_argument(
        '--rank',
        type=int,
        default=0,
        help='DISTRIBUTED: process rank for distributed training.')
    parser.add_argument(
        '--group_id',
        type=str,
        default="",
        help='DISTRIBUTED: process group id.')
    args = parser.parse_args()

    # setup output paths and read configs
    c = load_config(args.config_path)
    _ = os.path.dirname(os.path.realpath(__file__))
    if args.data_path != '':
        c.data_path = args.data_path

    if args.output_path == '':
        OUT_PATH = os.path.join(_, c.output_path)
    else:
        OUT_PATH = args.output_path

    if args.group_id == '' and args.output_folder == '':
        OUT_PATH = create_experiment_folder(OUT_PATH, c.run_name, args.debug)
    else:
        OUT_PATH = os.path.join(OUT_PATH, args.output_folder)

    AUDIO_PATH = os.path.join(OUT_PATH, 'test_audios')

    if args.rank == 0:
        os.makedirs(AUDIO_PATH, exist_ok=True)
        new_fields = {}
        if args.restore_path:
            new_fields["restore_path"] = args.restore_path
        new_fields["github_branch"] = get_git_branch()
        copy_config_file(args.config_path, os.path.join(OUT_PATH, 'config.json'), new_fields)
        os.chmod(AUDIO_PATH, 0o775)
        os.chmod(OUT_PATH, 0o775)

    if args.rank == 0:
        LOG_DIR = OUT_PATH
        tb_logger = Logger(LOG_DIR)

    try:
        main(args)
    except KeyboardInterrupt:
        remove_experiment_folder(OUT_PATH)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0) #pylint: disable=protected-access
    except Exception: #pylint: disable=broad-except
        remove_experiment_folder(OUT_PATH)
        traceback.print_exc()
        sys.exit(1)
