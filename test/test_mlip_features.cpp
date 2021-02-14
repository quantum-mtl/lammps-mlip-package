#include <gtest/gtest.h>

#include "mlip_features.h"
#include "mlip_polynomial.h"

TEST(MLIPFeaturesTest, cartesian_to_spherical) {
  auto v = vector1d{1, 1, 0};
  auto theta_phi = cartesian_to_spherical(v);
  ASSERT_DOUBLE_EQ(theta_phi[0], M_PI / 2.0);
  ASSERT_DOUBLE_EQ(theta_phi[1], M_PI / 4.0);
}

TEST(MLIPFeaturesTest, get_lm_info) {
  auto lm_info = get_lm_info(2);
  // list of (l, m, offset + id of m, offset + id of -m) where m < 0, offset = l * l
  vector2i expect;
  expect.emplace_back(vector1i{0,  0,     0,     0});
  expect.emplace_back(vector1i{1, -1, 1 + 0, 1 + 2});
  expect.emplace_back(vector1i{1,  0, 1 + 1, 1 + 1});
  expect.emplace_back(vector1i{2, -2, 4 + 0, 4 + 4});
  expect.emplace_back(vector1i{2, -1, 4 + 1, 4 + 3});
  expect.emplace_back(vector1i{2,  0, 4 + 2, 4 + 2});
  ASSERT_EQ(lm_info, expect);
}

TEST(MLIPFeaturesTest, get_ylm) {
  auto sph = vector1d{0, M_PI};
  auto lm_info = get_lm_info(2);
  vector1dc ylm;
  get_ylm(sph, lm_info, ylm);
  // sqrt{(2 * l + 1) / (4 * pi)} delta_{m, 0}
  auto expect = vector1dc{
    sqrt(1.0 / (4.0 * M_PI)),
    0,
    sqrt(3 / (4 * M_PI)),
    0,
    0,
    sqrt(5 / (4 * M_PI)),
  };
  ASSERT_EQ(ylm.size(), expect.size());
  for(size_t i = 0; i < expect.size(); ++i) {
    ASSERT_DOUBLE_EQ(real(ylm[i]), real(expect[i]));
    ASSERT_DOUBLE_EQ(imag(ylm[i]), imag(expect[i]));
  }
}

TEST(MLIPFeaturesTest, get_ylm_dtheta) {
  auto sph = vector1d{M_PI / 2, M_PI};
  auto lm_info = get_lm_info(1);
  vector1dc ylm, ylm_theta;
  get_ylm(sph, lm_info, ylm, ylm_theta);
  auto expect = vector1dc {
    0,
    0,
    -sqrt(3.0 / (4.0 * M_PI)),
  };
  ASSERT_EQ(ylm_theta.size(), expect.size());
  for(size_t i = 0; i < expect.size(); ++i) {
    ASSERT_DOUBLE_EQ(real(ylm_theta[i]) + 1, real(expect[i]) + 1);
    ASSERT_DOUBLE_EQ(imag(ylm_theta[i]) + 1, imag(expect[i]) + 1);
  }
}

// Broken test https://github.com/google/googletest/blob/master/docs/advanced.md#temporarily-disabling-tests
TEST(MLIPFeaturesTest, DISABLED_get_ylm_dtheta_singular) {
  auto sph = vector1d{0, M_PI};
  auto lm_info = get_lm_info(1);
  vector1dc ylm, ylm_theta;
  get_ylm(sph, lm_info, ylm, ylm_theta);
  auto expect = vector1dc {
    0,
    sqrt(1 / (8.0 * M_PI)),
    0,
  };
  ASSERT_EQ(ylm_theta.size(), expect.size());
  for(size_t i = 0; i < expect.size(); ++i) {
    ASSERT_DOUBLE_EQ(real(ylm_theta[i]) + 1, real(expect[i]) + 1);
    ASSERT_DOUBLE_EQ(imag(ylm_theta[i]) + 1, imag(expect[i]) + 1);
  }
}

TEST(MLIPFeaturesTest, get_fn) {
  double cutoff = 4;
  std::string pair_type = "gaussian";

  int n = 2;
  auto params = vector2d(n);  // [[beta, r_n]]
  params[0] = vector1d{log(2), 0};
  params[1] = vector1d{log(2), 2};

  feature_params fp;
  fp.cutoff = cutoff;
  fp.params = params;
  fp.pair_type = pair_type;

  double dis = 2;
  vector1d fn;
  get_fn(dis, fp, fn);
  auto fn_expect = vector1d{0.03125, 0.5};
  for(int i = 0; i < n; ++i) {
    ASSERT_DOUBLE_EQ(fn[i], fn_expect[i]);
  }
}
