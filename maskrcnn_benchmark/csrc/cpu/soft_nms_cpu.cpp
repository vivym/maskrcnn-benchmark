#include "cpu/vision.h"


template <typename scalar_t>
at::Tensor soft_nms_cpu_kernel(const at::Tensor& dets_t,
                          const at::Tensor& scores_t,
                          const float threshold,
                          const unsigned int method,
                          const float sigma,
                          const float min_score) {
  AT_ASSERTM(!dets_t.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores_t.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets_t.type() == scores_t.type(), "dets should have the same type as scores");

  if (dets_t.numel() == 0) {
    return at::empty({0}, dets_t.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets_t.select(1, 0).contiguous();
  auto y1_t = dets_t.select(1, 1).contiguous();
  auto x2_t = dets_t.select(1, 2).contiguous();
  auto y2_t = dets_t.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t + 1) * (y2_t - y1_t + 1);

  auto ndets = dets_t.size(0);
  at::Tensor selected_t = at::zeros({ndets}, dets_t.options().dtype(at::kByte).device(at::kCPU));
  auto order_t = at::arange(0, ndets, dets_t.options().dtype(at::kLong).device(at::kCPU));
  auto scores_cloned = scores_t.clone();

  auto selected = selected_t.data<uint8_t>();
  auto order = order_t.data<int64_t>();
  auto scores = scores_cloned.data<scalar_t>();
  auto x1 = x1_t.data<scalar_t>();
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();
  auto areas = areas_t.data<scalar_t>();

  int64_t n = ndets;
  for (int64_t _i = 0; _i < n; _i++) {
    auto i = order[_i];
    auto maxscore = scores[i];
    auto maxpos = _i;

    auto _pos = _i + 1;
    // find the box with max score
    while (_pos < n) {
      auto pos = order[_pos];
      if (maxscore < scores[pos]) {
        maxscore = scores[pos];
        maxpos = _pos;
      }
      ++_pos;
    }
    selected[order[maxpos]] = 1;
    std::swap(order[_i], order[maxpos]);
    i = order[_i];

    auto tx1 = x1[i];
    auto ty1 = y1[i];
    auto tx2 = x2[i];
    auto ty2 = y2[i];
    auto tarea = areas[i];

    _pos = _i + 1;
    while(_pos < n) {
      auto pos = order[_pos];
      auto _area = areas[pos];
      auto _x1 = x1[pos];
      auto _x2 = x2[pos];
      auto _y1 = y1[pos];
      auto _y2 = y2[pos];
      auto iw = std::min(tx2, _x2) - std::max(tx1, _x1) + 1;
      if (iw > 0) {
        auto ih = std::min(ty2, _y2) - std::max(ty1, _y1) + 1;
        if (ih > 0) {
          auto ua = static_cast<float>(tarea) + _area - iw * ih;
          auto ov = iw * ih / ua; // iou between max box and detection box

          float weight = 0;
          if (method == 1) { // linear
            if (ov > threshold) {
              weight = 1 - ov;
            } else {
              weight = 1;
            }
          } else if (method == 2) { // gaussian
            weight = std::exp(-(ov * ov) / sigma);
          } else {  // original NMS
            if (ov > threshold) {
              weight = 0;
            } else {
              weight = 1;
            }
          }
          auto score = weight * scores[pos];
          scores[pos] = score;

          // if box score falls below threshold, discard the box by swapping with last box update n
          if (score < min_score) {
            std::swap(order[_pos], order[n - 1]);
            pos = order[_pos];
            --n;
            --_pos;
          }
        }
      }
      ++_pos;
    }
  }

  return at::nonzero(selected_t == 1).squeeze(1);
}

at::Tensor soft_nms_cpu(const at::Tensor& dets,
                    const at::Tensor& scores,
                    const float threshold,
                    const unsigned int method,
                    const float sigma,
                    const float min_score) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(dets.type(), "soft_nms", [&] {
    result = soft_nms_cpu_kernel<scalar_t>(dets, scores, threshold, method, sigma, min_score);
  });
  return result;
}
