// these parts are a work in progress.hpp


// let's try a single sim version
template <class model, int width> struct heunwork {
  constexpr static nsvar = model::num_svar;
  float x1[nsvar * width], x2[nsvar * width], dx1[nsvar * width],
      dx2[nsvar * width], z[width], zscl[nsvar], dt;
  uint64_t seed[4];
  heunwork(float dt, uint64_t seed[4], float zscl[nsvar])
      : dt(dt), seed(seed), zscl(zscl) {}

  INLINE void load_svars(float *src, uint32_t svar_stride) {
#pragma clang loop unroll(full)
    for (uint8_t svar = 0; svar < nsvar; svar++)
      load<width>(x1 + svar * width, src + svar * svar_stride);
  }

  INLINE void save_svars(float *dst, uint32_t svar_stride) {
#pragma clang loop unroll(full)
    for (uint8_t svar = 0; svar < nsvar; svar++)
      load<width>(dst + svar * svar_stride, x1 + svar * width);
  }

  INLINE void predict() {
#pragma clang loop unroll(full)
    for (int svar = 0; svar < nsvar; svar++) {
      scaledrandn<width>(z, seed, zscl + svar);
      sheunpred<width>(x1 + svar * width, x2 + svar * width, dx1 + svar * width,
                       z, dt);
    }
  }

  INLINE void stage1(float *cx1, float *pj) {
    model::template dfun<width>(w.dx1, w.x1, cx1, pj); // cx.cx1+j, p+j*model::num_parm);
    w.predict();
    model::template adhoc<width>(w.x2);
  }

  INLINE void correct() {
#pragma clang loop unroll(full)
    for (int svar = 0; svar < nsvar; svar++) {
      scaledrandn<width>(z, seed, zscl + svar);
      sheuncorr<width>(x1 + svar * width, dx1 + svar * width,
                       dx2 + svar * width, z, dt);
    }
  }

  INLINE void stage2(float *cx2, float *pj) {
    model::template dfun<width>(w.dx2, w.x2, cx2, pj);
    w.correct();
    model::template adhoc<width>(w.x2);
    
  }
};

template <typename model, int width=8>
static void step1(
  const cx &cx, const conn &c,
  float *x, // (num_svar, num_node)
  float *y, // (num_svar, num_node)
  float z[model::num_svar], // (num_svar, )
  const float *p, // (num_node, num_parm)
  const bool p_varies_node,
  const uint32_t t0,
  const uint32_t nt,
  const float dt,
  uint64_t seed[4] // (4,)
)
{
  constexpr uint8_t nsvar = model::num_svar;
  const uint32_t num_node = cx.num_node, horizon = cx.num_time;

  for (uint32_t i=0; i<(model::num_svar * cx.num_node); i++)
    y[i] = 0.0f;

  for (uint32_t t=t0; t<(t0+nt); t++) {
    for (uint32_t i = 0; i < cx.num_node; i++) {
      cx_j(cx, conn, t);

      for (uint32_t j = 0; j < num_node; j+=width) {
        heunwork<model, width> w(dt, seed, zscl);
        w.load_svars(x+j, num_node);
        w.stage1(cx.cx1 + j, p + j*model::num_parm);
        w.stage2(cx.cx2+j, p+j*model::num_parm)
        w.save_svars(x+j, num_node);

        // XXX stopped here
        int write_time = i_time & (cx.num_time - 1);
        load<width>(cx.buf + i_node * horizon + write_time, x);
      }
    }
  }
  for (uint32_t i = 0; i < (model::num_svar * cx.num_node); i++)
    y[i] += x[i];
  }
  float rnt = 1.f / ((float) nt);
  for (uint32_t i=0; i<(model::num_svar * cx.num_node); i++)
    y[i] *= rnt;
}
