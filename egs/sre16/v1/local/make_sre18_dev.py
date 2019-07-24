#!/usr/bin/env python
import threading
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import argparse
import os
import sys
import traceback


def get_args():
    """ Get args from stdin.
    """

    parser = argparse.ArgumentParser(
        description="Create a tar file for fast DNN training.  Each of minibatch data will "
                    "saved to a separate numpy file within tar file.  The output file can be "
                    "accessed in sequential mode or in random access mode but the sequential "
                    "more is faster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    parser.add_argument("--sampling-frequency", dest="sampling_frequency", type=str, choices=["8k", "16k"], default="8k",
                        help="Sampling frequency of the input files.")

    parser.add_argument("sre18_dev_dir", type=str, help="Path to SRE18 development directory.")

    parser.add_argument("output_dir", type=str, help="Path to output directory.")

    print(' '.join(sys.argv))

    args = parser.parse_args()

    args = process_args(args)

    return args


def process_args(args):
    """ Process the options got from get_args()
    """

    if args.sre18_dev_dir == '' or not os.path.exists(args.sre18_dev_dir):
        raise Exception("The specified sre18_dev_dir '{0}' not exist.".format(args.sre18_dev_dir))

    return args


def write_to_wav_scp(wav_scp, sampling_frequency, extension, name, file_path):
    if sampling_frequency == "8k":
        if extension == 'sph':
            wav_scp.write('{utt} sph2pipe -f wav -p -c 1 {sph} |\n'.format(utt=name, sph=file_path))
        else:
            wav_scp.write('{utt} ffmpeg -i {flac} -f wav -ar 8000 - |\n'.format(utt=name, flac=file_path))
    else:
        if extension == 'sph':
            wav_scp.write('{utt} sph2pipe -f wav -p -c 1 {sph} | sox -t wav - -r 16k -t wav - |\n'.format(utt=name, sph=file_path))
        else:
            wav_scp.write('{utt} ffmpeg -i {flac} -f wav -ar 16000 - |\n'.format(utt=name, flac=file_path))


def process_files(args):
    sre18_dev_dir = args.sre18_dev_dir
    output_dir = os.path.join(args.output_dir, 'sre18_dev_enroll')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sampling_frequency = args.sampling_frequency
    # temp_dir = os.path.join(output_dir, 'tmp')
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)

    utt2spk = open(os.path.join(output_dir, 'utt2spk'), 'wt')
    wav_scp = open(os.path.join(output_dir, 'wav.scp'), 'wt')
    meta = open(os.path.join(sre18_dev_dir, 'docs/sre18_dev_enrollment.tsv'), 'rt')

    meta.readline()  # header line
    line = meta.readline()
    utt2fixedutt = {}
    while line:
        tokens = line.split("\t")
        model_id, segment_id, side = tokens
        segment_id = segment_id.split('.')[0]
        utt2spk.write("{spk}-{utt} {spk}\n".format(spk=model_id, utt=segment_id))
        utt2fixedutt[segment_id] = "{spk}-{utt}".format(spk=model_id, utt=segment_id)
        line = meta.readline()
    utt2spk.close()
    meta.close()

    audio_files = os.listdir(os.path.join(sre18_dev_dir, 'data/enrollment'))
    for file_name in audio_files:
        file_path = os.path.join(sre18_dev_dir, 'data/enrollment/' + file_name)
        name, extension = file_name.split('.')
        write_to_wav_scp(wav_scp, sampling_frequency, extension, utt2fixedutt[name], file_path)
    wav_scp.close()

    if os.system("utils/utt2spk_to_spk2utt.pl {out_dir}/utt2spk > {out_dir}/spk2utt".format(out_dir=output_dir)) != 0:
        raise Exception("Error creating spk2utt file in directory {out_dir}".format(out_dir=output_dir))

    if os.system("utils/fix_data_dir.sh {out_dir}".format(out_dir=output_dir)) != 0:
        raise Exception("Error fixing data dir {out_dir}".format(out_dir=output_dir))

    #################### TEST PART ####################
    output_dir = os.path.join(args.output_dir, 'sre18_dev_test')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # temp_dir = os.path.join(output_dir, 'tmp')
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)

    utt2spk = open(os.path.join(output_dir, 'utt2spk'), 'wt')
    wav_scp = open(os.path.join(output_dir, 'wav.scp'), 'wt')
    trials = open(os.path.join(output_dir, 'trials'), 'wt')
    trial_key = open(os.path.join(sre18_dev_dir, 'docs/sre18_dev_trial_key.tsv'), 'rt')
    segment_key = open(os.path.join(sre18_dev_dir, 'docs/sre18_dev_segment_key.tsv'), 'rt')

    segment_key.readline()  # header line
    line = segment_key.readline()
    utt2subject = {}
    while line:
        parts = line.split('\t')
        utt = parts[0].split('.')[0]
        subject = parts[1]
        utt2subject[utt] = subject
        line = segment_key.readline()

    audio_files = os.listdir(os.path.join(sre18_dev_dir, 'data/test'))
    for file_name in audio_files:
        file_path = os.path.join(sre18_dev_dir, 'data/test/' + file_name)
        name, extension = file_name.split('.')
        utt2spk.write('{utt} {utt}\n'.format(utt=name))
        write_to_wav_scp(wav_scp, sampling_frequency, extension, name, file_path)
    wav_scp.close()
    utt2spk.close()

    trial_key.readline()  # header line
    line = trial_key.readline()
    while line:
        parts = line.split('\t')
        spk = parts[0]
        utt = parts[1].split('.')[0]
        target_type = parts[3]
        trials.write('{spk} {utt} {target_type}\n'.format(spk=spk, utt=utt, target_type=target_type))
        line = trial_key.readline()
    trials.close()

    if os.system("utils/utt2spk_to_spk2utt.pl {out_dir}/utt2spk > {out_dir}/spk2utt".format(out_dir=output_dir)) != 0:
        raise Exception("Error creating spk2utt file in directory {out_dir}".format(out_dir=output_dir))

    if os.system("utils/fix_data_dir.sh {out_dir}".format(out_dir=output_dir)) != 0:
        raise Exception("Error fixing data dir {out_dir}".format(out_dir=output_dir))

    #################### UNLABELED PART ####################
    output_dir = os.path.join(args.output_dir, 'sre18_dev_unlabeled')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    utt2spk = open(os.path.join(output_dir, 'utt2spk'), 'wt')
    wav_scp = open(os.path.join(output_dir, 'wav.scp'), 'wt')

    audio_files = os.listdir(os.path.join(sre18_dev_dir, 'data/unlabeled'))
    for file_name in audio_files:
        file_path = os.path.join(sre18_dev_dir, 'data/unlabeled/' + file_name)
        name, extension = file_name.split('.')
        utt2spk.write('{utt} {utt}\n'.format(utt=name))
        write_to_wav_scp(wav_scp, sampling_frequency, extension, name, file_path)
    wav_scp.close()
    utt2spk.close()

    if os.system("utils/utt2spk_to_spk2utt.pl {out_dir}/utt2spk > {out_dir}/spk2utt".format(out_dir=output_dir)) != 0:
        raise Exception("Error creating spk2utt file in directory {out_dir}".format(out_dir=output_dir))

    if os.system("utils/fix_data_dir.sh {out_dir}".format(out_dir=output_dir)) != 0:
        raise Exception("Error fixing data dir {out_dir}".format(out_dir=output_dir))


def wait_for_background_commands():
    """ This waits for all threads to exit.  You will often want to
        run this at the end of programs that have launched background
        threads, so that the program will wait for its child processes
        to terminate before it dies."""
    for t in threading.enumerate():
        if not t == threading.current_thread():
            t.join()


def main():
    args = get_args()
    try:
        process_files(args)
        wait_for_background_commands()
    except BaseException as e:
        # look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
