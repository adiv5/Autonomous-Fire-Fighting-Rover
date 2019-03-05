/* LaHaShem HaAretz U'Mloah */

#ifndef HMM_FEATURES_H
#define HMM_FEATURES_H

#include "hmm-def.h"

typedef Real (*distance_metric)(Real, Real, Real);

/* A similar struct w.r.t c-optical-flow in scheme */
typedef struct{
  Real *flow;
  int hh;
  int ww;
} Flow;

typedef struct{
  char *name;
  int nn;
  Real *mean;
  Real *sigma;
  int dm_id;
} FeatureMedoid;

typedef struct{
  int p0[MAX_NTRACKS];
  int d0[3];
  int j0;
} M0;

struct FeaturesStruct {
  int tt;                     /* The moive length */
  int ii;                     /* The vector length in one frame */
  Real **v;
};

typedef struct FeaturesStruct *Features;

M0 *allocM0();

void set_sigmoid_width(Real width);

void set_dataset(char *set);

char* get_dataset();

/* Optical flow */
Flow *allocFlow(int ww, int hh);

void set_flow(Flow *fl, Real *flow, int ww, int hh);

void freeFlow(Flow *fl);

Real average_flow_in_box(Box *d, Flow *fl, Real scale, int flag);

Real average_flow_x_in_box(Box *d, Flow *fl, Real scale);

Real average_flow_y_in_box(Box *d, Flow *fl, Real scale);

Box *forward_projected_box_scaled(Box *d, Flow *fl, Real scale);

Real box_pair_score(Box *d1, Box *d2, Flow *fl, Real scale);

Real box_single_score(Box *d);

Real box_similarity(Box *d1, Box *d2);

/* FeatureMedoid */
FeatureMedoid *allocFeatureMedoid(int nn);

void freeFeatureMedoid(FeatureMedoid *fm);

void set_feature_medoid_name(FeatureMedoid *fm, char *name);

void set_feature_medoid_mean(FeatureMedoid *fm, int i, Real x);

void set_feature_medoid_sigma(FeatureMedoid *fm, int i, Real x);

void set_feature_medoid_dm_id(FeatureMedoid *fm, int id);

/* Features */
Features allocFeatures(int tt, int ii);

void freeFeatures(Features f);

int Features_tt(Features f);

int Features_ii(Features f);

Real **Features_v(Features f);

Real get_feature(Features f, int t, int i);

void print_Features(Features f);

/* Quantization functions */
int quantize(Real x, char *type, FeatureMedoid **fms, int mn);

int nearest_1d(Real x, Real *mean, Real *sigma, int nn, distance_metric dm);

Real fake_mahalanobis(Real x, Real mean, Real sigma);

/* from QobiScheme-AD.sc */
Real normalize_rotation(Real rotation);

Real angle_separation(Real x, Real y);

Real my_angle_separation(Real x, Real mean, Real sigma);

/*  Feature computation functions */
Features new_feature(enum part_of_speech pos, int d1, int d2, int d3, 
		     int **phi1, int ***phi2, int ***phi0, M0 *m0);

#endif
/* Tam V'Nishlam Shevah L'El Borei Olam */
