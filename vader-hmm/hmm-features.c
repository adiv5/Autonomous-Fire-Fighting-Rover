/* LaHaShem HaAretz U'Mloah */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "hmm.h"
#include "hmm-data.h"
#include "hmm-features.h"

#define CLIP(x, t, u) \
  if (x < t) x = t;    \
  if (x > u) x = u;

volatile double sigmoid_width = -1;
char dataset[100] = {0};

M0* allocM0(){
  M0 *m0 = safe_malloc(sizeof *m0);
  memset(m0->p0, -1, MAX_NTRACKS * sizeof *(m0->p0));
  memset(m0->d0, -1, 3 * sizeof *(m0->d0));
  m0->j0 = -1;
  return m0;
}

void set_sigmoid_width(Real width) { sigmoid_width = width; }

void set_dataset(char *set) { strcpy(dataset, set); }

char* get_dataset() { return dataset; }

static void panic (const char *error_text, ...)
{ 
  va_list ap;
  char error_message[1024];
  va_start(ap, error_text);
  vsprintf(error_message, error_text, ap);
  fprintf(stderr, "panic: %s\n", error_message);
  exit(EXIT_FAILURE);
}

Flow *allocFlow(int ww, int hh){
  Flow *fl = safe_malloc(sizeof *fl);
  fl->ww = ww;
  fl->hh = hh;
  fl->flow = safe_malloc(ww * hh * 2 * sizeof *(fl->flow));
  return fl;
}

void freeFlow(Flow *fl){
  if (fl){
    if (fl->flow){
      free(fl->flow);
    }
    free(fl);
  }
}

void set_flow(Flow *fl, Real *flow, int ww, int hh){
  assert(fl && fl->flow && flow);
  assert(fl->ww == ww && fl->hh == hh);
  memcpy(fl->flow, flow, ww * hh * 2 * sizeof(Real));
}

Real average_flow_x_in_box(Box *d, Flow *fl, Real scale){
  return average_flow_in_box(d, fl, scale, 0);
}

Real average_flow_y_in_box(Box *d, Flow *fl, Real scale){
  return average_flow_in_box(d, fl, scale, 1);
}

/* Can't include idealib-c.h */
Real integral_optical_flow_area(Real *integral_flow,
				const unsigned height,
				const unsigned width,
				unsigned x1, unsigned y1,
				unsigned x2, unsigned y2);

Real average_flow_in_box(Box *d, Flow *fl, Real scale, int flag){
  int x1, y1, x2, y2, area;
  Real integral = 0;
  assert(d && fl);
  /* The box dimension is 1280 x 960
     The videos used are 640 x 480
     The optical flow is 320 x 240
   */
  x1 = (int)round(d->x1 * scale / 2);
  y1 = (int)round(d->y1 * scale / 2);
  x2 = (int)round(d->x2 * scale / 2);
  y2 = (int)round(d->y2 * scale / 2);
  CLIP(x1, 0, fl->ww - 1);
  CLIP(y1, 0, fl->hh - 1);
  CLIP(x2, 0, fl->ww - 1);
  CLIP(y2, 0, fl->hh - 1);
  area = (x2 - x1 + 1) * (y2 - y1 + 1);

  switch(flag){
  case 0:  /* x direction */
    integral = integral_optical_flow_area(fl->flow, fl->hh, fl->ww,
					  x1, y1, x2, y2);
    break;
  case 1:
    integral = integral_optical_flow_area(fl->flow + fl->hh * fl->ww,
					  fl->hh, fl->ww,
					  x1, y1, x2, y2);
    break;
  default: panic("average_flow_in_box: flag = %d unrecognized", flag);
  }
  /* Because we downsize the square by half */
  return integral / area * 4;
}

Box *forward_projected_box_scaled(Box *d, Flow *fl, Real scale){
  Real shift_x = average_flow_x_in_box(d, fl, scale) / scale;
  Real shift_y = average_flow_y_in_box(d, fl, scale) / scale;
  Box *new_d = box_shift(d, shift_x, shift_y);
  return new_d;
}

/* Constants in these two functions are fixed empirically for all detections
   (i.e., the a and b arguments for sigmoid function)
*/
Real box_pair_score(Box *d1, Box *d2, Flow *fl, Real scale){
  Box *forwarded_d = forward_projected_box_scaled(d1, fl, scale);
  Real distance = box_distance(forwarded_d, d2);
  freeBox(forwarded_d);
  return sigmoid(distance, 50, - 1.0/11);
}

Real box_single_score(Box *d){
  /* In order to suppress detection response when the object is
     not in the field of view, the sigmoid means may need to be
     specified according to the dataset
   */
  Real a = 0, b = 6; // general case
  assert(sigmoid_width != -1);
  assert(dataset != NULL);

  if (strcmp(dataset, "new4") == 0){    
    b = sigmoid_width;    
    if (strcmp(d->model, "chair") == 0)
      a = -0.45;
    else if (strcmp(d->model, "stool") == 0)
      a = -0.60;
    else if (strcmp(d->model, "backpack") == 0)
      a = -0.5;
    else if (strcmp(d->model, "trash-can") == 0)
      a = -0.55;
    else if (strcmp(d->model, "traffic-cone") == 0)
      a = -0.65;
    else{
      // person
      a = -0.5;
      b = sigmoid_width / 2;
    }
  }
  return sigmoid(d->strength, a, b);
}

Real box_similarity(Box *d1, Box *d2){
  if (strcmp(d1->model, d2->model) == 0)
    return 1.0;
  return 0.0;
}

FeatureMedoid *allocFeatureMedoid(int nn){
  FeatureMedoid *fm = safe_malloc(sizeof *fm);
  fm->nn = nn;
  fm->mean = safe_malloc(nn * sizeof *(fm->mean));
  fm->sigma = safe_malloc(nn * sizeof *(fm->sigma));
  fm->name = NULL;
  return fm;
}

void set_feature_medoid_name(FeatureMedoid *fm, char *name){
  assert(fm && name);
  if (fm->name) free(fm->name);
  fm->name = safe_malloc(strlen(name) + 1);
  strcpy(fm->name, name);
}

void set_feature_medoid_mean(FeatureMedoid *fm, int i, Real x){
  assert(fm);
  if (i < 0 || i >= fm->nn)
    panic("setFeatureMedoidMean: i = %d out of range [0, %d]", i, fm->nn - 1);
  fm->mean[i] = x;
}

void set_feature_medoid_sigma(FeatureMedoid *fm, int i, Real x){
  assert(fm);
  if (i < 0 || i >= fm->nn)
    panic("setFeatureMedoidSigma: i = %d out of range [0, %d]", i, fm->nn - 1);
  if (x < 0.0)
    panic("setFeatureMedoidSigma: sigma %0.4lf negative", x);
  fm->sigma[i] = x;
}

void set_feature_medoid_dm_id(FeatureMedoid *fm, int id){
  assert(fm);
  if (id != 0 && id != 1)
    panic("set_feature_medoid_dm_id: id = %d unrecognized distance metric id", id);
  fm->dm_id = id;
}

void freeFeatureMedoid(FeatureMedoid *fm){
  if (fm){
    free(fm->mean);
    free(fm->sigma);
    if (fm->name) free(fm->name);
    free(fm);
  }
}

Features allocFeatures(int tt, int ii){
  int t;
  Features f = safe_malloc(sizeof *f);
  f->tt = tt;
  f->ii = ii;
  f->v = safe_malloc(tt * sizeof *(f->v));
  for (t = 0; t < tt; t ++)
    f->v[t] = safe_malloc(ii * sizeof *(f->v[t]));
  return f;
}

void freeFeatures(Features f){
  int t;
  if(f){
    if (f->v){
      for (t = 0; t < f->tt; t ++)
	free(f->v[t]);
      free(f->v);
    }
    free(f);
  }
}

int Features_tt(Features f){
  assert(f);
  return f->tt;
}

int Features_ii(Features f){
  assert(f);
  return f->ii;
}

Real **Features_v(Features f){
  assert(f);
  return f->v;
}

Real get_feature(Features f, int t, int i){
  assert(f);
  if (t < 0 || t >= f->tt)
    panic("get_feature: t = %d out of range [0, %d)", t, f->tt);
  if (i < 0 || i >= f->ii)
    panic("get_feature: i = %d out of range [0, %d)", i, f->ii);
  return f->v[t][i];
}

void print_Features(Features f){
  int t, i;
  assert(f);
  for (t = 0; t < f->tt; t ++){
    for (i = 0; i < f->ii; i ++)
      printf("%0.4lf ", f->v[t][i]);
    printf("\n");
  }
}

int quantize(Real x, char *type, FeatureMedoid **fms, int mn){
  int m, nn;
  double *mean = NULL, *sigma = NULL;
  distance_metric dm = NULL;
  for (m = 0; m < mn; m ++)
    if (strcmp(type, fms[m]->name) == 0)
      break;
  if (m == mn)
    panic("quantize: type = %s unrecognized value type!", type);
  nn = fms[m]->nn;
  mean = fms[m]->mean;
  sigma = fms[m]->sigma;
  dm = (fms[m]->dm_id == 0)? fake_mahalanobis: my_angle_separation;
  return nearest_1d(x, mean, sigma, nn, dm);
}

Real fake_mahalanobis(Real x, Real mean, Real sigma){
  return fabs(x - mean) / sqrt(sigma);
}

Real my_angle_separation(Real x, Real mean, Real sigma){
  return angle_separation(x, mean);
}

Real normalize_rotation(Real rotation){
  if (rotation > PI)
    return normalize_rotation(rotation - 2 * PI);
  else if (rotation <= -PI)
    return normalize_rotation(rotation + 2 * PI);
  else
    return rotation;
}

Real angle_separation(Real x, Real y){
  Real x_y, y_x;
  x_y = x - y;
  y_x = y - x;
  return MIN(fabs(normalize_rotation(x_y)), fabs(normalize_rotation(y_x)));
}

int nearest_1d(Real x, Real *mean, Real *sigma, int nn, distance_metric dm){
  Real minimum = INFINITY, distance;
  int n, m, position = -1;

  for (n = 0; n < nn; n ++){
    distance = dm(x, mean[n], sigma[n]);
    /* Find out whether there is another cluster beween x and mean[n]
       If ture, set the distance from x to mean[n] as INFINITY
     */
    for (m = 0; m < nn; m ++)
      if ((mean[n] < mean[m] && mean[m] < x)
	  || (x < mean[m] && mean[m] < mean[n]))
	break;
    if (dm != my_angle_separation && m < nn)
      distance = INFINITY;
    if (distance < minimum){
      minimum = distance;
      position = n;
    }
  }
  assert(position != -1);
  return position;
}

/* 
   phi1
   model_name
   velocity magnitude
   velocity orientation
   phi2
   // the same frame
   x-distance
   box ratio
   x-position
   phi0
   // the initial frame
   displacement magnitude
   displacement orientation
*/
Features new_feature(enum part_of_speech pos, int d1, int d2, int d3, 
		     int **phi1, int ***phi2, int ***phi0, M0 *m0){
  Features f = NULL;
  int i, d, d_anchor;
  switch(pos){
  case NOUN:
    /* Feature for a noun is the model name
       Only the first track should be valid */
    assert(d2 == -1 && d3 == -1);
    f = allocFeatures(1, 1);
    f->v[0][0] = phi1[d1][0];
    break;  
  case VERB:
    /* At least two tracks*/
    assert(d2 != -1);
    if (d3 == -1){
      /* Features for a verb with two participants are:
       1. agent velocity magnitude
       2. agent velocity orientation
       3. patient velocity magnitude
       4. patient velocity orientation
       5. agent-patient-x-distance
       6. agent-patient-relative-box-area
      */
      f = allocFeatures(1, 6);
      // 1 - 4
      for (i = 0; i < 4; i ++){
	if (i < 2) d = d1;
	else d = d2;       
	f->v[0][i] = phi1[d][i%2+1];
      }
      // 5
      f->v[0][4] = phi2[d1][d2][0];
      // 6
      f->v[0][5] = phi2[d1][d2][1];
    }
    else{
      /* Features for a verb with three participants are:
       1. agent velocity magnitude
       2. agent velocity orientation
       3. patient velocity magnitude
       4. patient velocity orientation
       5. goal velocity magnitude
       6. goal velocity orientation
       7. agent-patient-x-distance
       8. agent-patient-y-distance
       9. agent-patient-relative-box-area
       10. agent-patient-overlap
       11. agent-goal-x-distance
       12. agent-goal-y-distance
       13. agent-goal-relative-box-area
       14. agent-goal-overlap
       15. patient-goal-x-distance
       16. patient-goal-y-distance
       17. patient-goal-relative-box-area
       18. patient-goal-overlap
      */
    /*   f = allocFeatures(tt, 18); */
    /*   // 1 - 6 */
    /*   for (i = 0; i < 6; i ++){ */
    /* 	if (i < 2) tr = tr1; */
    /* 	else if (i < 4) tr = tr2; */
    /* 	else tr = tr3; */
    /* 	_raw = raw; */
    /* 	f1 = SINGLE_DETECTION_FEATURES(velocity_types[i % 2], tr); */
    /* 	for (t = 0; t < tt; t ++) */
    /* 	  f->v[t][i] = f1->v[t][0]; */
    /* 	freeFeatures(f1); */
    /*   } */
    /*   // 7 - 18 */
    /*   for (i = 6; i < 18; i += 4){ */
    /* 	if (i < 10){ */
    /* 	  tr = tr1; */
    /* 	  tr_ = tr2; */
    /* 	} */
    /* 	else if (i < 14){ */
    /* 	  tr = tr1; */
    /* 	  tr_ = tr3; */
    /* 	} */
    /* 	else{ */
    /* 	  tr = tr2; */
    /* 	  tr_ = tr3; */
    /* 	} */
    /* 	/\* x-distance *\/ */
    /* 	_raw = 1; */
    /* 	f1 = SINGLE_DETECTION_FEATURES(POSITION_X, tr); */
    /* 	f2 = SINGLE_DETECTION_FEATURES(POSITION_X, tr_); */
    /* 	for (t = 0; t < tt; t ++){ */
    /* 	  f->v[t][i] = DISTANCE_2D(f1->v[t][0], 0, f2->v[t][0], 0); */
    /* 	  if (raw == 0) */
    /* 	    f->v[t][i] = quantize(f->v[t][i], "x_distance", fms, mn); */
    /* 	} */
    /* 	freeFeatures(f1); */
    /* 	freeFeatures(f2); */
    /* 	/\* y-distance *\/ */
    /* 	_raw = 1; */
    /* 	f1 = SINGLE_DETECTION_FEATURES(POSITION_Y, tr); */
    /* 	f2 = SINGLE_DETECTION_FEATURES(POSITION_Y, tr_); */
    /* 	for (t = 0; t < tt; t ++){ */
    /* 	  f->v[t][i+1] = DISTANCE_2D(0, f1->v[t][0], 0, f2->v[t][0]); */
    /* 	  if (raw == 0) */
    /* 	    f->v[t][i+1] = quantize(f->v[t][i+1], "y_distance", fms, mn); */
    /* 	} */
    /* 	freeFeatures(f1); */
    /* 	freeFeatures(f2); */
    /* 	/\* relative-box-area *\/ */
    /* 	_raw = 1; */
    /* 	f1 = SINGLE_DETECTION_FEATURES(AREA, tr); */
    /* 	f2 = SINGLE_DETECTION_FEATURES(AREA, tr_); */
    /* 	for (t = 0; t < tt; t ++){ */
    /* 	  f->v[t][i+2] = f1->v[t][0] / f2->v[t][0]; */
    /* 	  if (raw == 0) */
    /* 	    f->v[t][i+2] = 1 - quantize(f->v[t][i+2], "box_ratio", fms, mn); */
    /* 	} */
    /* 	freeFeatures(f1); */
    /* 	freeFeatures(f2); */
    /* 	/\* overlap *\/ */
    /* 	for (t = 0; t < tt; t ++){ */
    /* 	  f->v[t][i+3] = box_overlap_percentage(tr->ds[t], tr_->ds[t]); */
    /* 	  if (raw == 0) */
    /* 	    f->v[t][i+3] = quantize(f->v[t][i+3], "overlap", fms, mn); */
    /* 	} */
    /*   } */
    } 
    break;
  case ADVERB:
    /* Features for an adverb is:
       agent velocity magnitude
     */
    assert(d2 == -1 && d3 == -1);
    f = allocFeatures(1, 1);
    f->v[0][0] = phi1[d1][1];
    break;
  case PREPOSITION:
    /* Feature for a preposition is the relative position of two objects:*/
    assert(d2 != -1 && d3 == -1);
    f = allocFeatures(1, 1);
    f->v[0][0] = phi2[d1][d2][2];
    break;
  case MOTION_PREPOSITION:
    /* Features for a motion_preposition are:
       1. agent velocity
       2. agent and location distance
    */
    assert(d2 != -1 && d3 == -1);
    f = allocFeatures(1, 2);
    f->v[0][0] = phi1[d1][1];
    f->v[0][1] = phi2[d1][d2][0];
    break;
  case VERB1:
    /* right now only support two tracks*/
    assert(d2 != -1 && d3 == -1);
    assert(phi0);
    /* Features for a verb with two participants are:
       1.  agent velocity magnitude
       2.  agent velocity orientation
       3.  patient velocity magnitude
       4.  patient velocity orientation
       5.  agent displacement magnitude
       6.  agent displacement orientation
       7.  patient displacement magnitude
       8.  patient displacement orientation
       9.  agent-patient-x-distance
       10. agent-patient-relative-box-area
    */
    f = allocFeatures(1, 10);
    // 1 - 4
    for (i = 0; i < 4; i ++){
      if (i < 2) d = d1;
      else d = d2;
      f->v[0][i] = phi1[d][i%2+1];
    }
    // 5 - 8
    for (i = 4; i < 8; i ++){
      if (i < 6){
	d = d1;
  	d_anchor = m0->d0[0];
      }
      else{
  	d = d2;
  	d_anchor = m0->d0[1];
      }      
      assert(d_anchor != -1);
      f->v[0][i] = phi0[d][d_anchor][i%2];
    }
    // 9
    f->v[0][8] = phi2[d1][d2][0];
    // 10
    f->v[0][9] = phi2[d1][d2][1];
    break;
  default: panic("word_features: part of speech %d unrecognized", pos);
  }
  return f;
}

/* Tam V'Nishlam Shevah L'El Borei Olam */
