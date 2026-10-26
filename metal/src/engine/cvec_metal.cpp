// metal/src/engine/cvec_metal.cpp - the control vector on Metal: --cvec-mode mlp only.  The loader's tables stay on the host
// (unified memory); the Window copies them into its own buffer the first time it encodes a layer, and adds layer l's d to
// the MLP output bo before the hyper-connection write (glue.metal's cvec_add_bo).  STRATA_METAL_CVEC_LIVE=FILE: every layer
// from 1 is steered from FILE instead (f32 [layers][n_embd], re-read when it changes), so a new vector needs no restart.  `covers()` stays false so the shared
// host paths written for the CUDA kernels (prefill, verify) never steer.
#include "strata/kernels/cvec.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::metal_engine {
namespace {
strata::kernels::Cvec g_cvec;
std::vector<float> g_dir;
std::vector<bool> g_layer;   // s_l != 0
bool g_on = false;
int* g_on_dev = nullptr;     // the Window's flag buffer, read by cvec_add_bo
}  // namespace

const std::vector<float>& cvec_dir() { return g_dir; }
const char* cvec_live() {
    static const char* p = std::getenv("STRATA_METAL_CVEC_LIVE");
    return p;
}
bool cvec_layer(int l) {
    if (cvec_live() && g_cvec.loaded()) return l >= 1 && l < (int) g_layer.size();
    return g_cvec.loaded() && l >= g_cvec.first && l <= g_cvec.last && l < (int) g_layer.size() && g_layer[(size_t) l]; }
void cvec_bind_flag(int* on) {
    g_on_dev = on;
    *g_on_dev = g_on ? 1 : 0;
}
}  // namespace strata::metal_engine

namespace strata::kernels {
using namespace strata::metal_engine;

const Cvec& cvec() { return g_cvec; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last, int64_t n_embd,
                 int64_t hc, std::string& err) {
    if (mode != 2) { err = "control vector: Metal implements --cvec-mode mlp only"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    g_dir.assign(dir.size(), 0.0f);
    g_layer.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) {
        g_layer[l] = s[l] != 0.0f;
        for (int64_t j = 0; j < n_embd; ++j) g_dir[l * (size_t) n_embd + (size_t) j] = s[l] * dir[l * (size_t) n_embd + (size_t) j];
    }
    g_on = true;
    if (g_on_dev) *g_on_dev = 1;
    g_cvec = Cvec{};
    g_cvec.dir = g_dir.data();
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    return true;
}

bool cvec_replicate(std::string&) { return true; }

void cvec_set_enabled(bool on) {
    if (!g_cvec.loaded()) return;
    g_on = on;
    if (g_on_dev) *g_on_dev = on ? 1 : 0;   // shared storage: read by the next command buffer (requests run one at a time)
}

bool cvec_enabled() { return g_cvec.loaded() && g_on; }

bool cvec_tables(const float**, const float**, const int**) { return false; }

void cvec_apply(float*, int64_t, int64_t, int64_t, const float*, int64_t, const float*, int64_t, bool, void*) {
    std::fprintf(stderr, "strata metal: cvec_apply is not used on Metal (--cvec-mode mlp steers inside the Window)\n");
}
}  // namespace strata::kernels
