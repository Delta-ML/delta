/* Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <limits>
#include "pitch.h"

namespace delta {

BaseFloat vector_Min(const std::vector<BaseFloat> &input) {
  BaseFloat min_res = std::numeric_limits<BaseFloat>::infinity();
  int length = input.size();
  int i;
  for (i = 0; i + 4 <= length; i += 4){
    BaseFloat a1 = input[i], a2 = input[i+1], a3 = input[i+2], a4 = input[i+3];
    if (a1 < min_res || a2 < min_res || a3 < min_res || a4 < min_res) {
      BaseFloat b1 = (a1 < a2 ? a1 : a2), b2 = (a3 < a4 ? a3 : a4);
      if (b1 < min_res) min_res = b1;
      if (b2 < min_res) min_res = b2;
    }
  }
  for(; i < length; i++){
    if (min_res > input[i]){
      min_res = input[i];
    }
  }
  return min_res;
}

BaseFloat vector_sum(const std::vector<BaseFloat> &input){
  BaseFloat res = 0;
  for (int i = 0; i < input.size(); i++)
    res += input[i];
  return res;
}

BaseFloat NccfToPovFeature(BaseFloat n) {
  if (n > 1.0) {
    n = 1.0;
  } else if (n < -1.0) {
    n = -1.0;
  }
  BaseFloat f = pow((1.0001 - n), 0.15) - 1.0;
  assert(f - f == 0);  // check for NaN,inf.
  return f;
}

BaseFloat NccfToPov(BaseFloat n) {
  BaseFloat ndash = abs(n);
  if (ndash > 1.0) ndash = 1.0;  // just in case it was slightly outside [-1, 1]

  BaseFloat r = -5.2 + 5.4 * exp(7.5 * (ndash - 1.0)) + 4.8 * ndash -
                2.0 * exp(-10.0 * ndash) + 4.2 * exp(20.0 * (ndash - 1.0));
  // r is the approximate log-prob-ratio of voicing, log(p/(1-p)).
  BaseFloat p = 1.0 / (1 + exp(-1.0 * r));
  assert(p - p == 0);  // Check for NaN/inf
  return p;
}

void ComputeCorrelation(const vector<BaseFloat> &wave,
                        int first_lag, int last_lag,
                        int nccf_window_size,
                        vector<BaseFloat> *inner_prod,
                        vector<BaseFloat> *norm_prod) {
  vector<BaseFloat> zero_mean_wave(wave);
  // TODO: possibly fix this, the mean normalization is done in a strange way.
  vector<BaseFloat> wave_part = sub_vector(wave, 0, nccf_window_size);
  // subtract mean-frame from wave
  BaseFloat wave_sum = 0.0;
  for (int i = 0; i < wave_part.size(); i++)
    wave_sum += wave_part[i];
  BaseFloat wave_mean = wave_sum / nccf_window_size;
  for (int j = 0; j < zero_mean_wave.size(); j++){
    zero_mean_wave[j] -= wave_mean;
  }
  BaseFloat e1, e2, sum;
  vector<BaseFloat> sub_vec1 = sub_vector(zero_mean_wave, 0, nccf_window_size);
  e1 = VecVec(sub_vec1, sub_vec1);
  for (int lag = first_lag; lag <= last_lag; lag++) {
    vector<BaseFloat> sub_vec2 = sub_vector(zero_mean_wave, lag, nccf_window_size);
    e2 = VecVec(sub_vec2, sub_vec2);
    sum = VecVec(sub_vec1, sub_vec2);
    (*inner_prod)[lag - first_lag] = sum;
    (*norm_prod)[lag - first_lag] = e1 * e2;
  }
}

void ComputeNccf(const vector<BaseFloat> &inner_prod,
                 const vector<BaseFloat> &norm_prod,
                 BaseFloat nccf_ballast,
                 vector<BaseFloat> *nccf_vec) {
  assert(inner_prod.size() == norm_prod.size() &&
               inner_prod.size() == nccf_vec->size());
  for (int lag = 0; lag < inner_prod.size(); lag++) {
    BaseFloat numerator = inner_prod[lag],
        denominator = pow(norm_prod[lag] + nccf_ballast, 0.5),
        nccf;
    if (denominator != 0.0) {
      nccf = numerator / denominator;
    } else {
      assert(numerator == 0.0);
      nccf = 0.0;
    }
    assert(nccf < 1.01 && nccf > -1.01);
    (*nccf_vec)[lag] = nccf;
  }
}

void SelectLags(const PitchExtractionOptions &opts,
                vector<BaseFloat> *lags) {
  // choose lags relative to acceptable pitch tolerance
  BaseFloat min_lag = 1.0 / opts.max_f0, max_lag = 1.0 / opts.min_f0;

  std::vector<BaseFloat> tmp_lags;
  for (BaseFloat lag = min_lag; lag <= max_lag; lag *= 1.0 + opts.delta_pitch)
    tmp_lags.push_back(lag);
  lags->resize(tmp_lags.size());
  for (int i = 0; i < tmp_lags.size(); i++)
    (*lags)[i] = tmp_lags[i];
}


void ComputeLocalCost(const vector<BaseFloat> &nccf_pitch,
                      const vector<BaseFloat> &lags,
                      const PitchExtractionOptions &opts,
                      vector<BaseFloat> *local_cost) {
  // from the paper, eq. 5, local_cost = 1 - Phi(t,i)(1 - soft_min_f0 L_i)
  // nccf is the nccf on this frame measured at the lags in "lags".
  assert(nccf_pitch.size() == local_cost->size() &&
               nccf_pitch.size() == lags.size());
  // for (int i = 0; i < local_cost->size(); i++)
  //   (*local_cost)[i] = 1.0;
  // for (int j = 0; j < local_cost->size(); j++)
  //   (*local_cost)[j] -= nccf_pitch[j];
  for (int k = 0; k < local_cost->size(); k++){
    (*local_cost)[k] = 1 - (1 - opts.soft_min_f0 * lags[k]) * nccf_pitch[k];
  }
}


class PitchFrameInfo {
 public:
  void Cleanup(PitchFrameInfo *prev_frame);

  void SetBestState(int best_state,
      std::vector<std::pair<int, BaseFloat> > &lag_nccf);

  int ComputeLatency(int max_latency);

  /// This function updates
  bool UpdatePreviousBestState(PitchFrameInfo *prev_frame);

  /// This constructor is used for frame -1; it sets the costs to be all zeros
  /// the pov_nccf's to zero and the backpointers to -1.
  explicit PitchFrameInfo(int num_states);

  /// This constructor is used for subsequent frames (not -1).
  PitchFrameInfo(PitchFrameInfo *prev);

  /// Record the nccf_pov value.
  ///  @param  nccf_pov     The nccf as computed for the POV computation (without ballast).
  void SetNccfPov(const vector<BaseFloat> &nccf_pov);

  void ComputeBacktraces(const PitchExtractionOptions &opts,
                         const vector<BaseFloat> &nccf_pitch,
                         const vector<BaseFloat> &lags,
                         const vector<BaseFloat> &prev_forward_cost,
                         std::vector<std::pair<int, int> > *index_info,
                         vector<BaseFloat> *this_forward_cost);
 private:
  struct StateInfo {
    /// The state index on the previous frame that is the best preceding state
    /// for this state.
    int backpointer;
    /// the version of the NCCF we keep for the POV computation (without the
    /// ballast term).
    BaseFloat pov_nccf;
    StateInfo(): backpointer(0), pov_nccf(0.0) { }
  };
  std::vector<StateInfo> state_info_;
  /// the state index of the first entry in "state_info"; this will initially be
  /// zero, but after cleanup might be nonzero.
  int state_offset_;

  /// The current best state in the backtrace from the end.
  int cur_best_state_;

  /// The structure for the previous frame.
  PitchFrameInfo *prev_info_;
};


// This constructor is used for frame -1; it sets the costs to be all zeros
// the pov_nccf's to zero and the backpointers to -1.
PitchFrameInfo::PitchFrameInfo(int num_states)
    :state_info_(num_states), state_offset_(0),
    cur_best_state_(-1), prev_info_(NULL) { }


bool pitch_use_naive_search = false;  // This is used in unit-tests.


PitchFrameInfo::PitchFrameInfo(PitchFrameInfo *prev_info):
    state_info_(prev_info->state_info_.size()), state_offset_(0),
    cur_best_state_(-1), prev_info_(prev_info) { }

void PitchFrameInfo::SetNccfPov(const vector<BaseFloat> &nccf_pov) {
  int num_states = nccf_pov.size();
  assert(num_states == state_info_.size());
  for (int i = 0; i < num_states; i++){
    state_info_[i].pov_nccf = nccf_pov[i];
  }
}

void PitchFrameInfo::ComputeBacktraces(
    const PitchExtractionOptions &opts,
    const vector<BaseFloat> &nccf_pitch,
    const vector<BaseFloat> &lags,
    const vector<BaseFloat> &prev_forward_cost_vec,
    std::vector<std::pair<int, int> > *index_info,
    vector<BaseFloat> *this_forward_cost_vec) {
  int num_states = nccf_pitch.size();

  vector<BaseFloat> local_cost(num_states, 0);
  ComputeLocalCost(nccf_pitch, lags, opts, &local_cost);

  const BaseFloat delta_pitch_sq = pow(Log(1.0 + opts.delta_pitch), 2.0),
      inter_frame_factor = delta_pitch_sq * opts.penalty_factor;

  const vector<BaseFloat> prev_forward_cost = prev_forward_cost_vec;
  // vector<BaseFloat> this_forward_cost = *this_forward_cost_vec;
  if (index_info->empty())
    index_info->resize(num_states);

  // make it a reference for more concise indexing.
  std::vector<std::pair<int, int> > &bounds = *index_info;

  if (pitch_use_naive_search) {
    // This branch is only taken in unit-testing code.
    for (int i = 0; i < num_states; i++) {
      BaseFloat best_cost = std::numeric_limits<BaseFloat>::infinity();
      int best_j = -1;
      for (int j = 0; j < num_states; j++) {
        BaseFloat this_cost = (j - i) * (j - i) * inter_frame_factor
            + prev_forward_cost[j];
        if (this_cost < best_cost) {
          best_cost = this_cost;
          best_j = j;
        }
      }
      (*this_forward_cost_vec)[i] = best_cost;
      state_info_[i].backpointer = best_j;
    }
  } else {
    int last_backpointer = 0;
    for (int i = 0; i < num_states; i++) {
      int start_j = last_backpointer;
      BaseFloat best_cost = (start_j - i) * (start_j - i) * inter_frame_factor
          + prev_forward_cost[start_j];
      int best_j = start_j;

      for (int j = start_j + 1; j < num_states; j++) {
        BaseFloat this_cost = (j - i) * (j - i) * inter_frame_factor
            + prev_forward_cost[j];
        if (this_cost < best_cost) {
          best_cost = this_cost;
          best_j = j;
        } else {  // as soon as the costs stop improving, we stop searching.
          break;  // this is a loose lower bound we're getting.
        }
      }
      state_info_[i].backpointer = best_j;
      (*this_forward_cost_vec)[i] = best_cost;
      bounds[i].first = best_j;  // this is now a lower bound on the
                                 // backpointer.
      bounds[i].second = num_states - 1;  // we have no meaningful upper bound
                                          // yet.
      last_backpointer = best_j;
    }

    for (int iter = 0; iter < num_states; iter++) {
      bool changed = false;
      if (iter % 2 == 0) {  // go backwards through the states
        last_backpointer = num_states - 1;
        for (int i = num_states - 1; i >= 0; i--) {
          int lower_bound = bounds[i].first,
              upper_bound = std::min(last_backpointer, bounds[i].second);
          if (upper_bound == lower_bound) {
            last_backpointer = lower_bound;
            continue;
          }
          BaseFloat best_cost = (*this_forward_cost_vec)[i];
          int best_j = state_info_[i].backpointer, initial_best_j = best_j;

          if (best_j == upper_bound) {
            last_backpointer = best_j;
            continue;
          }
          for (int j = upper_bound; j > lower_bound + 1; j--) {
            BaseFloat this_cost = (j - i) * (j - i) * inter_frame_factor
                + prev_forward_cost[j];
            if (this_cost < best_cost) {
              best_cost = this_cost;
              best_j = j;
            } else {
              if (best_j > j)
                break;  // this is a loose lower bound we're getting.
            }
          }
          bounds[i].second = best_j;
          if (best_j != initial_best_j) {
            (*this_forward_cost_vec)[i] = best_cost;
            state_info_[i].backpointer = best_j;
            changed = true;
          }
          last_backpointer = best_j;
        }
      } else {  // go forwards through the states.
        last_backpointer = 0;
        for (int i = 0; i < num_states; i++) {
          int lower_bound = std::max(last_backpointer, bounds[i].first),
              upper_bound = bounds[i].second;
          if (upper_bound == lower_bound) {
            last_backpointer = lower_bound;
            continue;
          }
          BaseFloat best_cost = (*this_forward_cost_vec)[i];
          int best_j = state_info_[i].backpointer, initial_best_j = best_j;

          if (best_j == lower_bound) {
            last_backpointer = best_j;
            continue;
          }
          // Below, we have j < upper_bound because we know we've already
          // evaluated that point.
          for (int j = lower_bound; j < upper_bound - 1; j++) {
            BaseFloat this_cost = (j - i) * (j - i) * inter_frame_factor
                + prev_forward_cost[j];
            if (this_cost < best_cost) {
              best_cost = this_cost;
              best_j = j;
            } else {
              if (best_j < j)
                break;  // this is a loose lower bound we're getting.
            }
          }
          // our "best_j" is now a lower bound on the backpointer.
          bounds[i].first = best_j;
          if (best_j != initial_best_j) {
            (*this_forward_cost_vec)[i] = best_cost;
            state_info_[i].backpointer = best_j;
            changed = true;
          }
          last_backpointer = best_j;
        }
      }
      if (!changed)
        break;
    }
  }
  // The next statement is needed due to RecomputeBacktraces: we have to
  // invalidate the previously computed best-state info.
  cur_best_state_ = -1;
  for (int i = 0; i < this_forward_cost_vec->size(); i++)
    (*this_forward_cost_vec)[i] += local_cost[i];
}

void PitchFrameInfo::SetBestState(
    int best_state,
    std::vector<std::pair<int, BaseFloat> > &lag_nccf) {

  std::vector<std::pair<int, BaseFloat> >::reverse_iterator iter = lag_nccf.rbegin();

  PitchFrameInfo *this_info = this;  // it will change in the loop.
  while (this_info != NULL) {
    PitchFrameInfo *prev_info = this_info->prev_info_;
    if (best_state == this_info->cur_best_state_)
      return;  // no change
    if (prev_info != NULL)  // don't write anything for frame -1.
      iter->first = best_state;
    size_t state_info_index = best_state - this_info->state_offset_;
    assert(state_info_index < this_info->state_info_.size());
    this_info->cur_best_state_ = best_state;
    best_state = this_info->state_info_[state_info_index].backpointer;
    if (prev_info != NULL)  // don't write anything for frame -1.
      iter->second = this_info->state_info_[state_info_index].pov_nccf;
    this_info = prev_info;
    if (this_info != NULL) ++iter;
  }
}

int PitchFrameInfo::ComputeLatency(int max_latency) {
  if (max_latency <= 0) return 0;

  int latency = 0;
  int num_states = state_info_.size();
  int min_living_state = 0, max_living_state = num_states - 1;
  PitchFrameInfo *this_info = this;  // it will change in the loop.


  for (; this_info != NULL && latency < max_latency;) {
    int offset = this_info->state_offset_;
    assert(min_living_state >= offset &&
                 max_living_state - offset < this_info->state_info_.size());
    min_living_state =
        this_info->state_info_[min_living_state - offset].backpointer;
    max_living_state =
        this_info->state_info_[max_living_state - offset].backpointer;
    if (min_living_state == max_living_state) {
      return latency;
    }
    this_info = this_info->prev_info_;
    if (this_info != NULL)  // avoid incrementing latency for frame -1,
      latency++;            // as it's not a real frame.
  }
  return latency;
}

void PitchFrameInfo::Cleanup(PitchFrameInfo *prev_frame) {
  std::cerr << "Cleanup not implemented.";
}

struct NccfInfo {

  vector<BaseFloat> nccf_pitch_resampled;  // resampled nccf_pitch
  BaseFloat avg_norm_prod; // average value of e1 * e2.
  BaseFloat mean_square_energy;  // mean_square energy we used when computing the
                                 // original ballast term for
                                 // "nccf_pitch_resampled".

  NccfInfo(BaseFloat avg_norm_prod,
           BaseFloat mean_square_energy):
      avg_norm_prod(avg_norm_prod),
      mean_square_energy(mean_square_energy) { }
};

class OnlinePitchFeatureImpl {
 public:
  explicit OnlinePitchFeatureImpl(const PitchExtractionOptions &opts);

  int Dim() const { return 2; }

  BaseFloat FrameShiftInSeconds() const;

  int NumFramesReady() const;

  bool IsLastFrame(int frame) const;

  void GetFrame(int frame, vector<BaseFloat> *feat);

  void AcceptWaveform(BaseFloat sampling_rate,
                      const vector<BaseFloat> &waveform);

  void InputFinished();

  ~OnlinePitchFeatureImpl();

  OnlinePitchFeatureImpl(const OnlinePitchFeatureImpl &other);

 private:

  int NumFramesAvailable(int num_downsampled_samples, bool snip_edges) const;
  void ExtractFrame(const vector<BaseFloat> &downsampled_wave_part,
                    int frame_index,
                    vector<BaseFloat> *window);

  void RecomputeBacktraces();

  void UpdateRemainder(const vector<BaseFloat> &downsampled_wave_part);
  PitchExtractionOptions opts_;

  // the first lag of the downsampled signal at which we measure NCCF
  int nccf_first_lag_;
  // the last lag of the downsampled signal at which we measure NCCF
  int nccf_last_lag_;

  // The log-spaced lags at which we will resample the NCCF
  vector<BaseFloat> lags_;
  ArbitraryResample *nccf_resampler_;
  LinearResample *signal_resampler_;

  std::vector<PitchFrameInfo*> frame_info_;
  std::vector<NccfInfo*> nccf_info_;

  int frames_latency_;
  vector<BaseFloat> forward_cost_;
  double forward_cost_remainder_;

  std::vector<std::pair<int, BaseFloat> > lag_nccf_;

  bool input_finished_;
  double signal_sumsq_;

  double signal_sum_;

  int downsampled_samples_processed_;
  vector<BaseFloat> downsampled_signal_remainder_;
};


OnlinePitchFeatureImpl::OnlinePitchFeatureImpl(
    const PitchExtractionOptions &opts):
    opts_(opts), forward_cost_remainder_(0.0), input_finished_(false),
    signal_sumsq_(0.0), signal_sum_(0.0), downsampled_samples_processed_(0) {
  signal_resampler_ = new LinearResample(opts.samp_freq, opts.resample_freq,
                                         opts.lowpass_cutoff,
                                         opts.lowpass_filter_width);

  double outer_min_lag = 1.0 / opts.max_f0 -
      (opts.upsample_filter_width/(2.0 * opts.resample_freq));
  double outer_max_lag = 1.0 / opts.min_f0 +
      (opts.upsample_filter_width/(2.0 * opts.resample_freq));
  nccf_first_lag_ = ceil(opts.resample_freq * outer_min_lag);
  nccf_last_lag_ = floor(opts.resample_freq * outer_max_lag);

  frames_latency_ = 0;  // will be set in AcceptWaveform()

  // Choose the lags at which we resample the NCCF.
  SelectLags(opts, &lags_);
  BaseFloat upsample_cutoff = opts.resample_freq * 0.5;


  vector<BaseFloat> lags_offset(lags_);
  for (int i = 0; i < lags_offset.size(); i++)
    lags_offset[i] += (-nccf_first_lag_ / opts.resample_freq);

  int num_measured_lags = nccf_last_lag_ + 1 - nccf_first_lag_;

  nccf_resampler_ = new ArbitraryResample(num_measured_lags, opts.resample_freq,
                                          upsample_cutoff, lags_offset,
                                          opts.upsample_filter_width);

  // add a PitchInfo object for frame -1 (not a real frame).
  frame_info_.push_back(new PitchFrameInfo(lags_.size()));
  // zeroes forward_cost_; this is what we want for the fake frame -1.
  forward_cost_.resize(lags_.size());
}


int OnlinePitchFeatureImpl::NumFramesAvailable(
    int num_downsampled_samples, bool snip_edges) const {
  int frame_shift = opts_.NccfWindowShift(),
      frame_length = opts_.NccfWindowSize();
  // Use the "full frame length" to compute the number
  // of frames only if the input is not finished.
  if (!input_finished_)
    frame_length += nccf_last_lag_;
  if (num_downsampled_samples < frame_length) {
    return 0;
  } else {
    if (!snip_edges) {
      if (input_finished_) {
        return static_cast<int>(num_downsampled_samples * 1.0f /
                                  frame_shift + 0.5f);
      } else {
        return static_cast<int>((num_downsampled_samples - frame_length / 2) *
                                   1.0f / frame_shift + 0.5f);
      }
    } else {
      return static_cast<int>((num_downsampled_samples - frame_length) /
                                 frame_shift + 1);
    }
  }
}

void OnlinePitchFeatureImpl::UpdateRemainder(
    const vector<BaseFloat> &downsampled_wave_part) {
  int num_frames = static_cast<int>(frame_info_.size()) - 1,
      next_frame = num_frames,
      frame_shift = opts_.NccfWindowShift(),
      next_frame_sample = frame_shift * next_frame;

  signal_sumsq_ += VecVec(downsampled_wave_part, downsampled_wave_part);
  BaseFloat downsampled_wave_part_sum = 0.0;
  for (int i = 0; i < downsampled_wave_part.size(); i++)
    downsampled_wave_part_sum += downsampled_wave_part[i];
  signal_sum_ += downsampled_wave_part_sum;

  int next_downsampled_samples_processed =
      downsampled_samples_processed_ + int(downsampled_wave_part.size());

  if (next_frame_sample > next_downsampled_samples_processed) {
    // this could only happen in the weird situation that the full frame length
    // is less than the frame shift.
    int full_frame_length = opts_.NccfWindowSize() + nccf_last_lag_;
    assert(full_frame_length < frame_shift);
    downsampled_signal_remainder_.resize(0);
  } else {
    vector<BaseFloat> new_remainder(next_downsampled_samples_processed -
                                    next_frame_sample);
    for (int i = next_frame_sample;
         i < next_downsampled_samples_processed; i++) {
      if (i >= downsampled_samples_processed_) {  // in current signal.
        new_remainder[i - next_frame_sample] =
            downsampled_wave_part[i - downsampled_samples_processed_];
      } else {  // in old remainder; only reach here if waveform supplied is
        new_remainder[i - next_frame_sample] =                      //  tiny.
            downsampled_signal_remainder_[i - downsampled_samples_processed_ +
                                          int(downsampled_signal_remainder_.size())];
      }
    }
    downsampled_signal_remainder_.swap(new_remainder);
  }
  downsampled_samples_processed_ = next_downsampled_samples_processed;
}

void OnlinePitchFeatureImpl::ExtractFrame(
    const vector<BaseFloat> &downsampled_wave_part,
    int sample_index,
    vector<BaseFloat> *window) {
  int full_frame_length = window->size();
  int offset = static_cast<int>(sample_index -
                                    downsampled_samples_processed_);
  // Treat edge cases first
  if (sample_index < 0) {
    assert(opts_.snip_edges == false);
    int sub_frame_length = sample_index + full_frame_length;
    int sub_frame_index = full_frame_length - sub_frame_length;
    assert(sub_frame_length > 0 && sub_frame_index > 0);
    for (int i = 0; i < window->size(); i++)
      (*window)[i] = 0.0;
    vector<BaseFloat> sub_window = sub_vector(*window, sub_frame_index, sub_frame_length);
    ExtractFrame(downsampled_wave_part, 0, &sub_window);
    return;
  }

  if (offset + full_frame_length > int(downsampled_wave_part.size())) {
    assert(input_finished_);
    int sub_frame_length = int(downsampled_wave_part.size()) - offset;
    assert(sub_frame_length > 0);
    for (int i = 0; i < window->size(); i++)
      (*window)[i] = 0.0;
    vector<BaseFloat> sub_window = sub_vector(*window, 0, sub_frame_length);
    ExtractFrame(downsampled_wave_part, sample_index, &sub_window);
    return;
  }

  if (offset >= 0) {
    // frame is full inside the new part of the signal.
    for (int k = 0; k < full_frame_length; k++){
      (*window)[k] = downsampled_wave_part[offset + k];
    }
  } else {
    // frame is partly in the remainder and partly in the new part.
    int remainder_offset = int(downsampled_signal_remainder_.size()) + offset;
    assert(remainder_offset >= 0);  // or we didn't keep enough remainder.
    assert(offset + full_frame_length > 0);  // or we should have
                                                   // processed this frame last
                                                   // time.

    int old_length = -offset, new_length = offset + full_frame_length;
    for (int i = 0; i < old_length; i++){
      (*window)[i] = downsampled_signal_remainder_[remainder_offset + i];
    }
    for (int i = 0; i < new_length; i++){
      (*window)[i + old_length] = downsampled_wave_part[i];
    }
  }
  if (opts_.preemph_coeff != 0.0) {
    BaseFloat preemph_coeff = opts_.preemph_coeff;
    for (int i = int(window->size()) - 1; i > 0; i--)
      (*window)[i] -= preemph_coeff * (*window)[i-1];
    (*window)[0] *= (1.0 - preemph_coeff);
  }
}

bool OnlinePitchFeatureImpl::IsLastFrame(int frame) const {
  int T = NumFramesReady();
  assert(frame < T);
  return (input_finished_ && frame + 1 == T);
}

BaseFloat OnlinePitchFeatureImpl::FrameShiftInSeconds() const {
  return opts_.frame_shift_ms / 1000.0f;
}

int OnlinePitchFeatureImpl::NumFramesReady() const {
  int num_frames = lag_nccf_.size(),
      latency = frames_latency_;
  assert(latency <= num_frames);
  return num_frames - latency;
}


void OnlinePitchFeatureImpl::GetFrame(int frame,
                                      vector<BaseFloat> *feat) {
  assert(frame < NumFramesReady() && feat->size() == 2);
  (*feat)[0] = lag_nccf_[frame].second;
  (*feat)[1] = 1.0 / lags_[lag_nccf_[frame].first];
}

void OnlinePitchFeatureImpl::InputFinished() {
  input_finished_ = true;
  AcceptWaveform(opts_.samp_freq, vector<BaseFloat>());
  int num_frames = static_cast<size_t>(int(frame_info_.size()) - 1);
  if (num_frames < opts_.recompute_frame && !opts_.nccf_ballast_online)
    RecomputeBacktraces();
  frames_latency_ = 0;
}

void OnlinePitchFeatureImpl::RecomputeBacktraces() {
  assert(!opts_.nccf_ballast_online);
  int num_frames = static_cast<int>(frame_info_.size()) - 1;

  // The assertion reflects how we believe this function will be called.
  assert(num_frames <= opts_.recompute_frame);
  assert(nccf_info_.size() == static_cast<size_t>(num_frames));
  if (num_frames == 0)
    return;
  double num_samp = downsampled_samples_processed_, sum = signal_sum_,
      sumsq = signal_sumsq_, mean = sum / num_samp;
  BaseFloat mean_square = sumsq / num_samp - mean * mean;

  bool must_recompute = false;
  BaseFloat threshold = 0.01;
  for (int frame = 0; frame < num_frames; frame++)
    if (!ApproxEqual(nccf_info_[frame]->mean_square_energy,
                     mean_square, threshold))
      must_recompute = true;

  if (!must_recompute) {
    for (size_t i = 0; i < nccf_info_.size(); i++)
      delete nccf_info_[i];
    nccf_info_.clear();
    return;
  }

  int num_states = forward_cost_.size(),
      basic_frame_length = opts_.NccfWindowSize();

  BaseFloat new_nccf_ballast = pow(mean_square * basic_frame_length, 2) *
      opts_.nccf_ballast;

  double forward_cost_remainder = 0.0;
  vector<BaseFloat> forward_cost(num_states, 0),  // start off at zero.
      next_forward_cost(forward_cost);
  std::vector<std::pair<int, int > > index_info;

  for (int frame = 0; frame < num_frames; frame++) {
    NccfInfo &nccf_info = *nccf_info_[frame];
    BaseFloat old_mean_square = nccf_info_[frame]->mean_square_energy,
        avg_norm_prod = nccf_info_[frame]->avg_norm_prod,
        old_nccf_ballast = pow(old_mean_square * basic_frame_length, 2) *
            opts_.nccf_ballast,
        nccf_scale = pow((old_nccf_ballast + avg_norm_prod) /
                         (new_nccf_ballast + avg_norm_prod),
                         static_cast<BaseFloat>(0.5));
    for (int i = 0; i < nccf_info.nccf_pitch_resampled.size(); i++)
      nccf_info.nccf_pitch_resampled[i] *= nccf_scale;

    frame_info_[frame + 1]->ComputeBacktraces(
        opts_, nccf_info.nccf_pitch_resampled, lags_,
        forward_cost, &index_info, &next_forward_cost);

    forward_cost.swap(next_forward_cost);

    BaseFloat remainder = vector_Min(forward_cost);
    forward_cost_remainder += remainder;
    for (int i = 0; i < forward_cost.size(); i++)
      forward_cost[i] -= remainder;
  }

  forward_cost_remainder_ = forward_cost_remainder;
  forward_cost_.swap(forward_cost);

  int best_final_state;
  BaseFloat forward_cost_min = vector_Min(forward_cost_);
  int index_forward_cost_min = 0;
  while (forward_cost_[index_forward_cost_min] != forward_cost_min)
    index_forward_cost_min += 1;
  best_final_state = index_forward_cost_min;

  if (lag_nccf_.size() != static_cast<size_t>(num_frames))
    lag_nccf_.resize(num_frames);

  frame_info_.back()->SetBestState(best_final_state, lag_nccf_);
  frames_latency_ =
      frame_info_.back()->ComputeLatency(opts_.max_frames_latency);
  for (size_t i = 0; i < nccf_info_.size(); i++)
    delete nccf_info_[i];
  nccf_info_.clear();
}

OnlinePitchFeatureImpl::~OnlinePitchFeatureImpl() {
  delete nccf_resampler_;
  delete signal_resampler_;
  for (size_t i = 0; i < frame_info_.size(); i++)
    delete frame_info_[i];
  for (size_t i = 0; i < nccf_info_.size(); i++)
    delete nccf_info_[i];
}

void OnlinePitchFeatureImpl::AcceptWaveform(
    BaseFloat sampling_rate,
    const vector<BaseFloat> &wave) {
  // flush out the last few samples of input waveform only if input_finished_ ==
  // true.
  const bool flush = input_finished_;
  vector<BaseFloat> downsampled_wave;
  signal_resampler_->Resample(wave, flush, &downsampled_wave);
  double cur_sumsq = signal_sumsq_, cur_sum = signal_sum_;
  int cur_num_samp = downsampled_samples_processed_,
      prev_frame_end_sample = 0;
  if (!opts_.nccf_ballast_online) {
    cur_sumsq += VecVec(downsampled_wave, downsampled_wave);
    BaseFloat downsampled_wave_sum = 0.0;
    for (int j = 0; j < downsampled_wave.size(); j++)
      downsampled_wave_sum += downsampled_wave[j];
    cur_num_samp += downsampled_wave.size();
  }
  int end_frame = NumFramesAvailable(
      downsampled_samples_processed_ + downsampled_wave.size(), opts_.snip_edges);
  int start_frame = int(frame_info_.size()) - 1,
      num_new_frames = end_frame - start_frame;

  if (num_new_frames == 0) {
    UpdateRemainder(downsampled_wave);
    return;
  }

  int num_measured_lags = nccf_last_lag_ + 1 - nccf_first_lag_,
      num_resampled_lags = lags_.size(),
      frame_shift = opts_.NccfWindowShift(),
      basic_frame_length = opts_.NccfWindowSize(),
      full_frame_length = basic_frame_length + nccf_last_lag_;

  vector<BaseFloat> window(full_frame_length),
      inner_prod(num_measured_lags),
      norm_prod(num_measured_lags);

  vector<vector<BaseFloat>> nccf_pitch(num_new_frames, vector<BaseFloat>(num_measured_lags));
  vector<vector<BaseFloat>> nccf_pov(num_new_frames, vector<BaseFloat>(num_measured_lags));
  vector<BaseFloat> cur_forward_cost(num_resampled_lags);

  for (int frame = start_frame; frame < end_frame; frame++) {
    // start_sample is index into the whole wave, not just this part.
    int start_sample;
    if (opts_.snip_edges) {
      // Usual case: offset starts at 0
      start_sample = static_cast<int>(frame) * frame_shift;
    } else {
      start_sample =
        static_cast<int>((frame + 0.5) * frame_shift) - full_frame_length / 2;
    }
    ExtractFrame(downsampled_wave, start_sample, &window);
    if (opts_.nccf_ballast_online) {
      int end_sample = start_sample + full_frame_length -
          downsampled_samples_processed_;
      assert(end_sample > 0);  // or should have processed this frame last
                                     // time.  Note: end_sample is one past last
                                     // sample.
      if (end_sample > downsampled_wave.size()) {
        assert(input_finished_);
        end_sample = downsampled_wave.size();
      }
      vector<BaseFloat> new_part = sub_vector(downsampled_wave, prev_frame_end_sample,
                                    end_sample - prev_frame_end_sample);
      cur_num_samp += int(new_part.size());
      cur_sumsq += VecVec(new_part, new_part);
      BaseFloat new_part_sum = 0;
      for (int k = 0; k < new_part.size(); k++)
        new_part_sum += new_part[k];
      cur_sum += new_part_sum;
      prev_frame_end_sample = end_sample;
    }
    double mean_square = cur_sumsq / cur_num_samp -
        pow(cur_sum / cur_num_samp, 2.0);
    ComputeCorrelation(window, nccf_first_lag_, nccf_last_lag_,
                       basic_frame_length, &inner_prod, &norm_prod);
    BaseFloat norm_prod_sum = 0.0;
    for (int k = 0; k < norm_prod.size(); k++)
      norm_prod_sum += norm_prod[k];
    double nccf_ballast_pov = 0.0,
        nccf_ballast_pitch = pow(mean_square * basic_frame_length, 2) *
             opts_.nccf_ballast,
        avg_norm_prod = norm_prod_sum / int(norm_prod.size());
    ComputeNccf(inner_prod, norm_prod, nccf_ballast_pitch,
                &nccf_pitch[frame - start_frame]);
    ComputeNccf(inner_prod, norm_prod, nccf_ballast_pov,
                &nccf_pov[frame - start_frame]);
    if (frame < opts_.recompute_frame)
      nccf_info_.push_back(new NccfInfo(avg_norm_prod, mean_square));
  }

  vector<vector<BaseFloat>> nccf_pitch_resampled(num_new_frames, vector<BaseFloat>(num_resampled_lags));
  nccf_resampler_->Resample(nccf_pitch, &nccf_pitch_resampled);
  for (int i = 0; i < nccf_pitch.size(); i++)
    vector<BaseFloat>().swap(nccf_pitch[i]);
  nccf_pitch.clear();
  vector<vector<BaseFloat>> nccf_pov_resampled(num_new_frames, vector<BaseFloat>(num_resampled_lags));
  nccf_resampler_->Resample(nccf_pov, &nccf_pov_resampled);
  for (int i = 0; i < nccf_pov.size(); i++)
    vector<BaseFloat>().swap(nccf_pov[i]);
  nccf_pov.clear();
  UpdateRemainder(downsampled_wave);

  std::vector<std::pair<int, int > > index_info;

  for (int frame = start_frame; frame < end_frame; frame++) {
    int frame_idx = frame - start_frame;
    PitchFrameInfo *prev_info = frame_info_.back(),
        *cur_info = new PitchFrameInfo(prev_info);
    cur_info->SetNccfPov(nccf_pov_resampled[frame_idx]);
    cur_info->ComputeBacktraces(opts_, nccf_pitch_resampled[frame_idx],
                                lags_, forward_cost_, &index_info,
                                &cur_forward_cost);

    forward_cost_.swap(cur_forward_cost);
    BaseFloat remainder = vector_Min(forward_cost_);
    forward_cost_remainder_ += remainder;
    for (int k = 0; k < forward_cost_.size(); k++)
      forward_cost_[k] -= remainder;
    frame_info_.push_back(cur_info);
    if (frame < opts_.recompute_frame)
      nccf_info_[frame]->nccf_pitch_resampled =
          nccf_pitch_resampled[frame_idx];
    if (frame == opts_.recompute_frame - 1 && !opts_.nccf_ballast_online)
      RecomputeBacktraces();
  }

  int best_final_state;
  BaseFloat forward_cost_min2 = vector_Min(forward_cost_);
  for (int g = 0; g < forward_cost_.size(); g++){
    if (forward_cost_[g] == forward_cost_min2)
      best_final_state = g;
  }
  lag_nccf_.resize(int(frame_info_.size()) - 1);  // will keep any existing data.
  frame_info_.back()->SetBestState(best_final_state, lag_nccf_);
  frames_latency_ =
      frame_info_.back()->ComputeLatency(opts_.max_frames_latency);
}

int OnlinePitchFeature::NumFramesReady() const {
  return impl_->NumFramesReady();
}

OnlinePitchFeature::OnlinePitchFeature(const PitchExtractionOptions &opts)
    :impl_(new OnlinePitchFeatureImpl(opts)) { }

bool OnlinePitchFeature::IsLastFrame(int frame) const {
  return impl_->IsLastFrame(frame);
}

BaseFloat OnlinePitchFeature::FrameShiftInSeconds() const {
  return impl_->FrameShiftInSeconds();
}

void OnlinePitchFeature::GetFrame(int frame, vector<BaseFloat> *feat) {
  impl_->GetFrame(frame, feat);
}

void OnlinePitchFeature::AcceptWaveform(
    BaseFloat sampling_rate,
    const vector<BaseFloat> &waveform) {
  impl_->AcceptWaveform(sampling_rate, waveform);
}

void OnlinePitchFeature::InputFinished() {
  impl_->InputFinished();
}

OnlinePitchFeature::~OnlinePitchFeature() {
  delete impl_;
}

void ComputeKaldiPitchFirstPass(
    const PitchExtractionOptions &opts,
    const vector<BaseFloat> &wave,
    vector<vector<BaseFloat>> *output) {

  int cur_rows = 100;
  vector<vector<BaseFloat>> feats(cur_rows, vector<BaseFloat>(2));

  OnlinePitchFeature pitch_extractor(opts);
  assert(opts.frames_per_chunk > 0 );

  int cur_offset = 0, cur_frame = 0, samp_per_chunk =
      opts.frames_per_chunk * opts.samp_freq * opts.frame_shift_ms / 1000.0f;

  while (cur_offset < wave.size()) {
    int num_samp = std::min(samp_per_chunk, int(wave.size()) - cur_offset);
    vector<BaseFloat> wave_chunk = sub_vector(wave, cur_offset, num_samp);
    pitch_extractor.AcceptWaveform(opts.samp_freq, wave_chunk);
    cur_offset += num_samp;
    if (cur_offset == wave.size())
      pitch_extractor.InputFinished();
    // Get each frame as soon as it is ready.
    for (; cur_frame < pitch_extractor.NumFramesReady(); cur_frame++) {
      if (cur_frame >= cur_rows) {
        cur_rows *= 2;
        feats.resize(cur_rows);
        for(int i = 0; i < cur_rows; i++)
          feats[i].resize(2);
      }
      vector<BaseFloat> row(feats[cur_frame]);
      pitch_extractor.GetFrame(cur_frame, &row);
    }
  }
  if (cur_frame  == 0) {
    std::cerr << "No features output since wave file too short";
    for (int i = 0; i < output->size(); i++)
      vector<BaseFloat>().swap((*output)[i]);
    (*output).clear();
  } else {
    for(int j = 0; j < cur_frame; j++)
      (*output)[j] = feats[j];
  }
}

void ComputeKaldiPitch(const PitchExtractionOptions &opts,
                       const vector<BaseFloat> &wave,
                       vector<vector<BaseFloat>> *output) {
  if (opts.simulate_first_pass_online) {
    ComputeKaldiPitchFirstPass(opts, wave, output);
    return;
  }
  OnlinePitchFeature pitch_extractor(opts);

  if (opts.frames_per_chunk == 0) {
    pitch_extractor.AcceptWaveform(opts.samp_freq, wave);
  } else {
    // the user may set opts.frames_per_chunk for better compatibility with
    // online operation.
    assert(opts.frames_per_chunk > 0);
    int cur_offset = 0, samp_per_chunk =
        opts.frames_per_chunk * opts.samp_freq * opts.frame_shift_ms / 1000.0f;
    while (cur_offset < wave.size()) {
      int num_samp = std::min(samp_per_chunk, int(wave.size()) - cur_offset);
      vector<BaseFloat> wave_chunk = sub_vector(wave, cur_offset, num_samp);
      pitch_extractor.AcceptWaveform(opts.samp_freq, wave_chunk);
      cur_offset += num_samp;
    }
  }

  pitch_extractor.InputFinished();
  int num_frames = pitch_extractor.NumFramesReady();
  if (num_frames == 0) {
    std::cerr << "No frames output in pitch extraction";
    for (int i = 0; i < output->size(); i++)
      vector<BaseFloat>().swap((*output)[i]);
    (*output).clear();
    return;
  }

  output->resize(num_frames);
  for(int i = 0; i < num_frames; i++)
    (*output)[i].resize(2);
  for (int frame = 0; frame < num_frames; frame++) {
    pitch_extractor.GetFrame(frame, &((*output)[frame]));
  }
}

OnlineProcessPitch::OnlineProcessPitch(
    const ProcessPitchOptions &opts,
    OnlineFeatureInterface *src):
    opts_(opts), src_(src),
    dim_ ((opts.add_pov_feature ? 1 : 0)
          + (opts.add_normalized_log_pitch ? 1 : 0)
          + (opts.add_delta_pitch ? 1 : 0)
          + (opts.add_raw_log_pitch ? 1 : 0)) {
  assert(dim_ > 0);
  assert(src->Dim() == kRawFeatureDim);
}


void OnlineProcessPitch::GetFrame(int frame,
                                  vector<BaseFloat> *feat) {
  int frame_delayed = frame < opts_.delay ? 0 : frame - opts_.delay;
  assert(feat->size() == dim_ && frame_delayed < NumFramesReady());
  int index = 0;
  if (opts_.add_pov_feature)
    (*feat)[index++] = GetPovFeature(frame_delayed);
  if (opts_.add_normalized_log_pitch)
    (*feat)[index++] = GetNormalizedLogPitchFeature(frame_delayed);
  if (opts_.add_delta_pitch)
    (*feat)[index++] = GetDeltaPitchFeature(frame_delayed);
  if (opts_.add_raw_log_pitch)
    (*feat)[index++] = GetRawLogPitchFeature(frame_delayed);
  assert(index == dim_);
}

BaseFloat OnlineProcessPitch::GetPovFeature(int frame) const {
  vector<BaseFloat> tmp(kRawFeatureDim);
  src_->GetFrame(frame, &tmp);
  BaseFloat nccf = tmp[0];
  return opts_.pov_scale * NccfToPovFeature(nccf)
      + opts_.pov_offset;
}

BaseFloat OnlineProcessPitch::GetDeltaPitchFeature(int frame) {
  int context = opts_.delta_window;
  int start_frame = std::max(0, frame - context),
      end_frame = std::min(frame + context + 1, src_->NumFramesReady()),
      frames_in_window = end_frame - start_frame;
  vector<vector<BaseFloat>> feats(frames_in_window, vector<BaseFloat>(1)),
      delta_feats(frames_in_window, vector<BaseFloat>(2));

  for (int f = start_frame; f < end_frame; f++)
    feats[f - start_frame][0] = GetRawLogPitchFeature(f);

  DeltaDelta delta;
  delta.Initialize(1, opts_.delta_window);
  for (int i = 0; i < frames_in_window; i++)
    delta.Compute_frame(feats, frame, &(delta_feats[i]));

  while (delta_feature_noise_.size() <= static_cast<size_t>(frame)) {
    delta_feature_noise_.push_back(RandGauss() *
                                   opts_.delta_pitch_noise_stddev);
  }
  // note: delta_feats will have two columns, second contains deltas.
  return (delta_feats[frame - start_frame][1] + delta_feature_noise_[frame]) *
      opts_.delta_pitch_scale;
}

BaseFloat OnlineProcessPitch::GetRawLogPitchFeature(int frame) const {
  vector<BaseFloat> tmp(kRawFeatureDim);
  src_->GetFrame(frame, &tmp);
  BaseFloat pitch = tmp[1];
  assert(pitch > 0);
  return log(pitch);
}

BaseFloat OnlineProcessPitch::GetNormalizedLogPitchFeature(int frame) {
  UpdateNormalizationStats(frame);
  BaseFloat log_pitch = GetRawLogPitchFeature(frame),
      avg_log_pitch = normalization_stats_[frame].sum_log_pitch_pov /
        normalization_stats_[frame].sum_pov,
      normalized_log_pitch = log_pitch - avg_log_pitch;
  return normalized_log_pitch * opts_.pitch_scale;
}


// inline
void OnlineProcessPitch::GetNormalizationWindow(int t,
                                                int src_frames_ready,
                                                int *window_begin,
                                                int *window_end) const {
  int left_context = opts_.normalization_left_context;
  int right_context = opts_.normalization_right_context;
  *window_begin = std::max(0, t - left_context);
  *window_end = std::min(t + right_context + 1, src_frames_ready);
}

void OnlineProcessPitch::UpdateNormalizationStats(int frame) {
  assert(frame >= 0);
  if (normalization_stats_.size() <= frame)
    normalization_stats_.resize(frame + 1);
  int cur_num_frames = src_->NumFramesReady();
  bool input_finished = src_->IsLastFrame(cur_num_frames - 1);

  NormalizationStats &this_stats = normalization_stats_[frame];
  if (this_stats.cur_num_frames == cur_num_frames &&
      this_stats.input_finished == input_finished) {
    return;
  }
  int this_window_begin, this_window_end;
  GetNormalizationWindow(frame, cur_num_frames,
                         &this_window_begin, &this_window_end);

  if (frame > 0) {
    const NormalizationStats &prev_stats = normalization_stats_[frame - 1];
    if (prev_stats.cur_num_frames == cur_num_frames &&
        prev_stats.input_finished == input_finished) {
      this_stats = prev_stats;
      int prev_window_begin, prev_window_end;
      GetNormalizationWindow(frame - 1, cur_num_frames,
                             &prev_window_begin, &prev_window_end);
      if (this_window_begin != prev_window_begin) {
        assert(this_window_begin == prev_window_begin + 1);
        vector<BaseFloat> tmp(kRawFeatureDim);
        src_->GetFrame(prev_window_begin, &tmp);
        BaseFloat accurate_pov = NccfToPov(tmp[0]),
            log_pitch = log(tmp[1]);
        this_stats.sum_pov -= accurate_pov;
        this_stats.sum_log_pitch_pov -= accurate_pov * log_pitch;
      }
      if (this_window_end != prev_window_end) {
        assert(this_window_end == prev_window_end + 1);
        vector<BaseFloat> tmp(kRawFeatureDim);
        src_->GetFrame(prev_window_end, &tmp);
        BaseFloat accurate_pov = NccfToPov(tmp[0]),
            log_pitch = log(tmp[1]);
        this_stats.sum_pov += accurate_pov;
        this_stats.sum_log_pitch_pov += accurate_pov * log_pitch;
      }
      return;
    }
  }
  this_stats.cur_num_frames = cur_num_frames;
  this_stats.input_finished = input_finished;
  this_stats.sum_pov = 0.0;
  this_stats.sum_log_pitch_pov = 0.0;
  vector<BaseFloat> tmp(kRawFeatureDim);
  for (int f = this_window_begin; f < this_window_end; f++) {
    src_->GetFrame(f, &tmp);
    BaseFloat accurate_pov = NccfToPov(tmp[0]),
        log_pitch = log(tmp[1]);
    this_stats.sum_pov += accurate_pov;
    this_stats.sum_log_pitch_pov += accurate_pov * log_pitch;
  }
}

int OnlineProcessPitch::NumFramesReady() const {
  int src_frames_ready = src_->NumFramesReady();
  if (src_frames_ready == 0) {
    return 0;
  } else if (src_->IsLastFrame(src_frames_ready - 1)) {
    return src_frames_ready + opts_.delay;
  } else {
    return std::max(0, src_frames_ready -
      opts_.normalization_right_context + opts_.delay);
  }
}

void ProcessPitch(const ProcessPitchOptions &opts,
                  const vector<vector<BaseFloat>> &input,
                  vector<vector<BaseFloat>> *output) {
  OnlineMatrixFeature pitch_feat(input);

  OnlineProcessPitch online_process_pitch(opts, &pitch_feat);

  output->resize(online_process_pitch.NumFramesReady());
  for(int i = 0; i < online_process_pitch.NumFramesReady(); i++)
    (*output)[i].resize(online_process_pitch.Dim());
  for (int t = 0; t < online_process_pitch.NumFramesReady(); t++) {
    online_process_pitch.GetFrame(t, &((*output)[t]));
  }
}

}  // namespace delta
