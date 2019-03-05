/* LaHaShem HaAretz U'Mloah */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "hmm.h"
#include "hmm-rand.h"
#include "hmm-data.h"
#include "hmm-control.h"

static RVec **g_logL = NULL;
static RVec **g_gamma = NULL;
static RVec **g_gamma_r = NULL;
static int g_ll;
static int g_uu;
static RMat *g_lastData = NULL;

#define U(j) VECTOR(ed)[j]
#define J(k,j) VECTOR(ed2[j])[k]
#define W(u) VECTOR(es)[u]
#define V(v,u) VECTOR(es2[u])[v]
#define X(v,u) VECTOR(es3[u])[v]
#define SENTENCE_HMM_A(s, w) \
  s->ws[w]->hmm->A
#define SENTENCE_HMM_B(s, w) \
  s->ws[w]->hmm->B
#define OUTPUT_MODEL(s, w, u) \
  s->ws[w]->ffm[u]
#define SENTENCE_FFM_P(s, w, u, i) \
  s->ws[w]->ffm[u]->p[i]

/* may be changed by Scheme code */
volatile int MIN_DURATION = 8;
volatile int DURATION = FALSE;
volatile int NO_PRIOR = FALSE;
volatile int PAD_ALL = FALSE;
volatile int DISPLACEMENT = FALSE;
volatile enum model_constraint model_constraint = NO_MODEL_CONSTRAINT;
volatile enum state_constraint state_constraint = NO_STATE_CONSTRAINT;

/* scheme-c */
static void panic (const char *error_text, ...)
{ 
  va_list ap;
  char error_message[1024];
  va_start(ap, error_text);
  vsprintf(error_message, error_text, ap);
  fprintf(stderr, "panic: %s\n", error_message);
  exit(EXIT_FAILURE);
}

void set_min_duration(int new_min_duration){
  MIN_DURATION = new_min_duration;
}

int get_min_duration(){
  return MIN_DURATION;
}

void switch_sentence_prior(int bprior){
  NO_PRIOR = bprior;
}

void switch_duration(int bvalue){
  DURATION = bvalue;
}

void switch_displacement(int bvalue){
  DISPLACEMENT = bvalue;
}

int get_displacement(){
  return DISPLACEMENT;
}

int get_duration(){
  return DURATION;
}

int get_pad_all(){
  return PAD_ALL;
}

void set_model_constraint(enum model_constraint new_constraint){
  model_constraint = new_constraint;
}

void set_state_constraint(enum state_constraint new_constraint){
  state_constraint = new_constraint;
}

/* -- utility functions -- */
void initGlobals(int ll, int uu, RMat *data, int endmode)
{ int l, u;
  if (g_logL!=NULL)
  { for (l = 0; l<g_ll; l++)
    { for (u = 0; u<g_uu; u++)
      { freeRVec(g_logL[l][u]);
	freeRVec(g_gamma[l][u]);}
      free(g_logL[l]);
      free(g_gamma[l]);}
    for (u = 0; u<g_uu; u++) free(g_gamma_r[u]);
    free(g_logL);
    free(g_gamma);
    free(g_gamma_r);}
  g_logL = safe_malloc(sizeof *g_logL * ll);
  g_gamma = safe_malloc(sizeof *g_gamma * ll);
  g_gamma_r = safe_malloc(sizeof *g_gamma_r * uu);
  for (l = 0; l<ll; l++)
    { g_logL[l] = safe_malloc(sizeof *(g_logL[l]) * uu);
      g_gamma[l] = safe_malloc(sizeof *(g_gamma[l]) * uu);
    for (u = 0; u<uu; u++)
    { g_logL[l][u] = allocRVec(COLUMNS(data[l])+endmode);
      g_gamma[l][u] = allocRVec(COLUMNS(data[l])+endmode);}}
  for (u = 0; u<uu; u++)
    { g_gamma_r[u] = safe_malloc(sizeof *(g_gamma_r[u]) * ll);
      for (l = 0; l<ll; l++) g_gamma_r[u][l] = g_gamma[l][u];}
  g_uu = uu;
  g_ll = ll;}

/* allocModel - Allocates memory for a model with uu states and ii features. */
Model *allocModel(int ii, int uu){ 
  Model *m;
  int u;
  if (ii<=0||uu<=0) panic("allocModel(): %s must be>0", ii<=0?"ii":"uu");
  m = safe_malloc(sizeof *m);
  m->ffm = safe_malloc(uu * sizeof *(m->ffm));
  m->uu = uu;
  m->hmm = allocHMM(uu);
  for (u = 0; u<uu; u++) m->ffm[u] = allocFFM(ii);
  m->n_roles = 0;
  m->name = NULL;
  m->pos = OTHER;
  return m;
}

Model *cloneModel(Model *src_m){
  Model *m;
  m = allocModel(model_ii(src_m), model_uu(src_m));
  copyModel(m, src_m);
  return m;
}

/* copyModel - Copy the contents of one model to another */
void copyModel(Model *dst_m, Model *src_m) {
  int u;
  assert(src_m&&dst_m);
  assert(src_m->ffm[0]->ii==dst_m->ffm[0]->ii);

  if(src_m->uu != dst_m->uu){
    freeModel(dst_m);
    dst_m = allocModel(src_m->ffm[0]->ii, src_m->uu);
  }
  copyHMM(dst_m->hmm, src_m->hmm);
  for (u = 0; u<src_m->uu; u++)
    copyFFM(dst_m->ffm[u], src_m->ffm[u]);
  if (dst_m->name) free(dst_m->name);
  dst_m->name = safe_malloc(strlen(src_m->name) + 1);
  strcpy(dst_m->name, src_m->name);
  dst_m->n_roles = src_m->n_roles;
  dst_m->pos = src_m->pos;
}

void uniformModel(Model *m, int ut){
  int u;
  constantHMM(m->hmm, ut);
  for (u = 0; u < m->uu; u ++)
    constantParams(m->ffm[u]);
}

void randomiseModel(Model *m, int ut)
{ int u;
  randomiseHMM(m->hmm, ut);
  for (u = 0; u < m->uu; u++)
    randomiseParams(m->ffm[u]);
}

void noiseModel(Model *m, int upper_triangular, Real delta){
  int u;
  noiseHMM(m->hmm, upper_triangular, delta);
  for (u = 0; u < m->uu; u ++)
    noiseParams(m->ffm[u], delta);
}

void smoothModel(Model *m, int upper_triangular, Real eps){
  int u;
  smoothHMM(m->hmm, upper_triangular, eps);
  for (u = 0; u < m->uu; u ++)
    smoothParams(m->ffm[u], eps);
}

int normalizeModel(Model *m, int upper_triangular){
  assert(m);
  int u;
  if (normaliseHMM(m->hmm, upper_triangular) == FALSE){
    return FALSE;
  }
  for (u = 0; u < m->uu; u ++)
    if (normalizeFFM(m->ffm[u]) == FALSE)
      return FALSE;
  return TRUE;
}

/* defineContFeat - Defines feature i to be a continuous feature whose
 *  value is in the range (-sigma, sigma] with p = 0.68.  Non-zero mean
 *  features should be translated appropiately.
 */
void defineContFeat(Model *m, int i, Real sigma)
{ struct ContFI *fi;
  int u, ii = m->ffm[0]->ii;
  if (i<0||i>=ii)
  { panic("defineContFeat(): i = %d is outside range [0,%d]", i, ii-1);}
  if (sigma<=0.0) panic("defineContFeat(): sigma = %g must be>0.0", sigma);
  for (u = 0; u<m->uu; u++)
  { fi = safe_malloc(sizeof *fi);
    fi->initialSigma = sigma;
    setFeatType(m->ffm[u], i, FT_CONTINUOUS, (void *)fi);}}

/* defineRadialFeat - Defines feature i to be a radial feature whose
 *  value is in the range [0, 2*pi).
 */
void defineRadialFeat(Model *m, int i)
{ int u, ii = m->ffm[0]->ii;
  if (i<0||i>=ii)
  { panic("defineRadialFeat(): i = %d is outside range [0,%d]", i, ii-1);}
  for (u = 0; u<m->uu; u++)
  { setFeatType(m->ffm[u], i, FT_RADIAL, NULL);}}

/* defineDiscreteFeat - Defines feature i to be a discrete feature whose
 *  value is in the range [0, n-1].
 */
void defineDiscreteFeat(Model *m, int i)
{ int u, ii = m->ffm[0]->ii;
  if (i<0||i>=ii)
  { panic("defineDiscreteFeat(): i = %d is outside range [0,%d]", i, ii-1);}
  for (u = 0; u<m->uu; u++)
  { setFeatType(m->ffm[u], i, FT_DISCRETE, NULL);}}

void displayModel(void *p, Model *m)
{ int u;
  FILE *f = (FILE *)p;
  fprintf(f, "========== Model Display ========\n");
  fprintf(f, "Model name: %s\n", m->name);
  fprintf(f, "Model role number (%d):\n", m->n_roles);
  fprintf(f, "\nModel part_of_speech: %s\n", model_pos_str(m->pos));
  displayHMM(f, m->hmm);
  for (u = 0; u<m->uu; u++)
  { fprintf(f, "-------\n");
    fprintf(f, "U = %3d\n", u);
    fprintf(f, "-------\n\n");
    displayFFM(f, m->ffm[u]);}}

void print_model(Model *m) {displayModel(stdout, m);}

void freeModel(Model *m)
{ int u;
  freeHMM(m->hmm);
  for (u = 0; u<m->uu; u++) freeFFM(m->ffm[u]);
  free(m->ffm);
  if (m->name) free(m->name);
  free(m);}

int isZeroModel(Model *m){
  assert(m);
  int u;
  if (isZeroHMM(m->hmm) == 0)
    return 0;
  for (u = 0; u < m->uu; u ++)
    if (isZeroFFM(m->ffm[u]) == 0)
      return 0;
  return 1;
}

void zeroModel(Model *m){
  assert(m);
  int u;
  zeroHMM(m->hmm);
  for (u = 0; u < m->uu; u ++)
    zeroFFM(m->ffm[u]);
}

void linear2logModel(Model *m){
  assert(m);
  int u;
  linear2logHMM(m->hmm);
  for (u = 0; u < m->uu; u ++)
    linear2logFFM(m->ffm[u]);
}

void log2linearModel(Model *m){
  assert(m);
  int u;
  log2linearHMM(m->hmm);
  for (u = 0; u < m->uu; u ++)
    log2linearFFM(m->ffm[u]);
}

Sentence initializeSentence(int ww, Model **ws){
  Sentence s = safe_malloc(sizeof *s);
  s->ww = ww;
  s->ws = safe_malloc(ww * sizeof *(s->ws));
  int w;
  for (w = 0; w < ww; w ++)
    s->ws[w] = ws[w];
  return s;
}

void freeSentence(Sentence s){
  free(s->ws);
  free(s);
}

/* -- processing functions -- */

/* logLike -
 *  Calculates log likelihood of entire model given the feature data.
 */
Real logLike(Model *m, RMat data){ 
  int u, uu = m->uu;
  RVec *logL;
  Real rv;
  logL = safe_malloc(uu * sizeof *logL);
  for (u = 0; u<uu; u++)
    { assert(ROWS(data)==m->ffm[u]->ii);
      logL[u] = allocRVec(COLUMNS(data));
      FFM_logL(m->ffm[u], logL[u], data);}
  rv = HMM_logL(m->hmm, logL);
  for (u = 0; u<uu; u++) freeRVec(logL[u]);
  free(logL);
  /* needs work: Why is rv divided by the number of features? */
  return rv/model_ii(m);}

/* logLike_with_box_scores -
 *  Calculates log likelihood of entire model given the feature data.
 */
Real logLike_with_box_scores(Model *m, RMat data, RMat score_mat, int endmode)
{ int u, uu = m->uu;
  RVec *logL;
  Real rv;
  Real **scores = MATRIX(score_mat);
  logL = safe_malloc(uu * sizeof *logL);
  for (u = 0; u<uu; u++){
    logL[u] = allocRVec(COLUMNS(data)+endmode);
    FFM_logL_with_box_scores(m->ffm[u], logL[u], data,scores[u],u+1==uu,endmode);}
  rv = HMM_logL(m->hmm, logL);
  for (u = 0; u<uu; u++) freeRVec(logL[u]);
  free(logL);
  /* needs work: Why is rv divided by the number of features? */
  return rv/model_ii(m);}

/* best_state_sequence -
 *  Calculates the best state sequence of entire model given the feature data.
 */
int *best_state_sequence(Model *m, RMat data){ 
  int u, uu = m->uu, *rv;
  RVec *logL;
  logL = safe_malloc(uu * sizeof *logL);
  for (u = 0; u<uu; u++)
  { assert(ROWS(data)==m->ffm[u]->ii);
    logL[u] = allocRVec(COLUMNS(data));
    FFM_logL(m->ffm[u], logL[u], data);}
  rv = HMM_best_state_sequence(m->hmm, logL);
  for (u = 0; u<uu; u++) freeRVec(logL[u]);
  free(logL);
  return rv;}

double** state_probabilities(Model *m, RMat data){ 
  int u, uu = m->uu, t;
  RVec *gammas;
  RVec *logL;
  double **rv;
  logL = safe_malloc(uu * sizeof *logL);
  gammas = safe_malloc(uu * sizeof *gammas);
  rv = safe_malloc(COLUMNS(data) * sizeof *rv);
  for (u = 0; u<uu; u++)
  { assert(ROWS(data)==m->ffm[u]->ii);
    logL[u] = allocRVec(COLUMNS(data));
    gammas[u] = allocRVec(COLUMNS(data));

    FFM_logL(m->ffm[u], logL[u], data);}
  HMM_calcGammas(m->hmm,logL,gammas);

  for (t=0; t<COLUMNS(data); t++)
    { rv[t] = safe_malloc(uu * sizeof *(rv[t]));
    for(u=0; u<uu; u++)
     {
       rv[t][u] = VECTOR(gammas[u])[t];}}
  // copy the data into the array which is going to be passed to scheme
  for (u = 0; u<uu; u++)
   { freeRVec(logL[u]);
     freeRVec(gammas[u]);}
  free(logL);
  free(gammas);
  return rv;}

double** state_probabilities_with_box_scores(Model *m, RMat data,
					     RMat score_mat, int endmode){
  int u, uu = m->uu, t;
  RVec *gammas;
  RVec *logL;
  double **rv;
  Real **scores = MATRIX(score_mat);
  logL = safe_malloc(uu * sizeof *logL);
  gammas = safe_malloc(uu * sizeof *gammas);
  rv = safe_malloc((COLUMNS(data)+endmode) * sizeof *rv);
  for (u = 0; u<uu; u++)
  { assert(ROWS(data)==m->ffm[u]->ii);
    logL[u] = allocRVec(COLUMNS(data)+endmode);
    gammas[u] = allocRVec(COLUMNS(data)+endmode);

    FFM_logL_with_box_scores(m->ffm[u], logL[u], data,scores[u],u+1==uu,endmode);}
  HMM_calcGammas(m->hmm,logL,gammas);

  for (t=0; t<COLUMNS(data)+endmode; t++)
    { rv[t] = safe_malloc(uu * sizeof *(rv[t]));
    for(u=0; u<uu; u++)
     {
       rv[t][u] = VECTOR(gammas[u])[t];}}
  // copy the data into the array which is going to be passed to scheme
  for (u = 0; u<uu; u++)
   { freeRVec(logL[u]);
     freeRVec(gammas[u]);}
  free(logL);
  free(gammas);
  return rv;}


void force_init_globals(void) {g_lastData = NULL;}

void compute_posterior(Model **m, RMat *data, Real *prior, int *c_ls, int ll,
		       int cc, enum training_mode training_mode,
		       /* outputs */
		       Real *objective_function, Real *auxiliary, Real **postpC) {
  int l, c;
  Real postp_sum;
  assert(postpC);
  for (l = 0; l<ll; l++)
    for (c = 0; c<cc; c++)
      postpC[c][l] = logLike(m[c], data[l])+my_log(prior[c]);

  *objective_function = 0.0;
  *auxiliary = 0.0;
  for (l = 0; l<ll; l++) {
    postp_sum = NEGATIVE_INFINITY;
    for (c = 0; c<cc; c++) postp_sum = add_exp(postp_sum, postpC[c][l]);
    for (c = 0; c<cc; c++) {
	switch (training_mode) {
	case HMM_ML:
	  if(c==c_ls[l]) *objective_function += postpC[c][l];
	  if(postp_sum!=NEGATIVE_INFINITY)
	    postpC[c][l] -= postp_sum;
	  if(c==c_ls[l]) *auxiliary += postpC[c][l];
	  break;
	case HMM_DT:
	  if(c==c_ls[l]) *auxiliary += postpC[c][l];
	  if(postp_sum!=NEGATIVE_INFINITY)
	    postpC[c][l] -= postp_sum;
	  if(c==c_ls[l]) *objective_function += postpC[c][l];
	  break;
	default: panic("unrecognized training mode");
	}
    }
  }
}

/* update - Performs a single update on the HMM model for the given data. */
int update(Model **m, Hmm **tmp_hmm, RMat *data, Real **postpC, Real log_D,
	   int *c_ls, int ll, int cc, enum training_mode training_mode,
	   int upper_triangular) {
  /* tmp_hmm[c] is scratch space. It must have at least as many states as
     m[c]->hmm. */
  int c, l, u;
  for (c = 0; c<cc; c++) {
    if (g_lastData!=data||ll>g_ll||m[c]->uu>g_uu) {
      g_lastData = data;
      initGlobals(ll, m[c]->uu, data,0);
    }
    zeroHMM(tmp_hmm[c]);
    for (l = 0; l<ll; l++) {
      switch (training_mode) {
      case HMM_ML:
	if (c!=c_ls[l]) continue;
	for (u = 0; u<m[c]->uu; u++) {
	  FFM_logL(m[c]->ffm[u], g_logL[l][u], data[l]);
	}
	HMM_updateModel(m[c]->hmm, tmp_hmm[c], g_logL[l], g_gamma[l], log_D,
			0.0, -1, -1, training_mode);
	break;
      case HMM_DT:
	for (u = 0; u<m[c]->uu; u++) {
	  FFM_logL(m[c]->ffm[u], g_logL[l][u], data[l]);
	}
	HMM_updateModel(m[c]->hmm, tmp_hmm[c], g_logL[l], g_gamma[l], log_D,
			my_exp(postpC[c][l]), c, c_ls[l], training_mode);
	break;
      default: panic("unrecognized training mode");
      }
    }
    if (!normaliseHMM(tmp_hmm[c], upper_triangular)) {
      assert(training_mode == HMM_DT);
      return FALSE;
    }
    copyHMM(m[c]->hmm, tmp_hmm[c]);
    for (u = 0; u<m[c]->uu; u++) {
      switch (training_mode) {
      case HMM_ML:
	assert(FFM_maximise(m[c]->ffm[u], data, g_gamma_r[u], ll, log_D,
			    NULL, c, c_ls, 0));
	break;
      case HMM_DT:
	if (!FFM_maximise(m[c]->ffm[u], data, g_gamma_r[u], ll, log_D,
			  postpC[c], c, c_ls, 0)){
	  return FALSE;
	}
	break;
      default: panic("unrecognized training mode");
      }
    }
  }
  return TRUE;
}

int update_with_box_scores(Model **m, Hmm **tmp_hmm, RMat *data, Real **postpC, Real log_D,
	   int *c_ls, int ll, int cc, enum training_mode training_mode,
			   int upper_triangular, RMat *score_matrices,int endmode) {
  /* tmp_hmm[c] is scratch space. It must have at least as many states as
     m[c]->hmm. */
  int c, l, u;
  Real** scores;
  for (c = 0; c<cc; c++) {
    if (g_lastData!=data||ll>g_ll||m[c]->uu>g_uu) {
      g_lastData = data;
      initGlobals(ll, m[c]->uu, data,endmode);
    }
    zeroHMM(tmp_hmm[c]);
    for (l = 0; l<ll; l++) {
      scores = MATRIX(score_matrices[l]);
      switch (training_mode) {
      case HMM_ML:
	if (c!=c_ls[l]) continue;
	for (u = 0; u<m[c]->uu; u++) {
	  FFM_logL_with_box_scores(m[c]->ffm[u], g_logL[l][u], data[l],scores[u],u+1==m[c]->uu,endmode);
	}
	HMM_updateModel(m[c]->hmm, tmp_hmm[c], g_logL[l], g_gamma[l], log_D,
			0.0, -1, -1, training_mode);
	break;
      case HMM_DT:
	for (u = 0; u<m[c]->uu; u++) {
	  FFM_logL_with_box_scores(m[c]->ffm[u], g_logL[l][u], data[l],scores[u],u+1==m[c]->uu,endmode);
	}
	HMM_updateModel(m[c]->hmm, tmp_hmm[c], g_logL[l], g_gamma[l], log_D,
			my_exp(postpC[c][l]), c, c_ls[l], training_mode);
	break;
      default: panic("unrecognized training mode");
      }
    }
    if (!normaliseHMM(tmp_hmm[c], upper_triangular)) {
      assert(training_mode == HMM_DT);
      return FALSE;
    }
    copyHMM(m[c]->hmm, tmp_hmm[c]);
    for (u = 0; u<m[c]->uu; u++) {
      switch (training_mode) {
      case HMM_ML:
	assert(FFM_maximise(m[c]->ffm[u], data, g_gamma_r[u], ll, log_D,
			    NULL, c, c_ls, endmode));
	break;
      case HMM_DT:
	if (!FFM_maximise(m[c]->ffm[u], data, g_gamma_r[u], ll, log_D,
			  postpC[c], c, c_ls, endmode))
	  return FALSE;
	break;
      default: panic("unrecognized training mode");
      }
    }
  }
  return TRUE;
}

void removeStates(Model *m, int *xu, Hmm *hmm)
{
  int v, u, w, old_uu = m->uu;
  Hmm *phmm = m->hmm;
  for(u = 0; u < m->uu; ){
    if(xu[u]){
      if(u < m->uu - 1){
	for(v = u; v < m->uu - 1; v ++){
	  xu[v] = xu[v+1];
	  copyFFM(m->ffm[v], m->ffm[v+1]);
	  set_model_b(m, v, VECTOR(phmm->B)[v+1]);
	  copyRVec(MATRIX(phmm->A)[v], MATRIX(phmm->A)[v+1], m->uu);
	  for(w = 0; w < m->uu; w ++)
	    set_model_a(m, w, v, MATRIX(phmm->A)[w][v+1]);
	  if (hmm){
	    VECTOR(hmm->B)[v] = VECTOR(hmm->B)[v+1];
	    copyRVec(MATRIX(hmm->A)[v], MATRIX(hmm->A)[v+1], m->uu);
	    for(w = 0; w < m->uu; w ++)
	      MATRIX(hmm->A)[w][v] = MATRIX(hmm->A)[w][v+1];
	  }
	}
      }
      m->uu --;
      phmm->uu --;
      if (hmm) hmm->uu --;
    }
    else u ++;
  }
  assert(m->uu > 0);
  /* Clean up the redundant memory. DO NOT clean up hmm matrix now. It will be released
     according to the size of RMat instead of hmm->uu. */
  for(u = m->uu; u < old_uu; u ++)
    freeFFM(m->ffm[u]);
}

/* --------------------- Sentence training stuff ---------------------*/

void decode_box_states(int j, int ii, int nTracks, int *js){
  int i;
  for (i = nTracks - 1; i >= 0; i --){
    js[i] = j % ii;
    j /= ii;
  }
}

/* log space */
void compute_sentence_priors(Sentence *ss, int nn, int tt, Real *priors){
  int n, w, i, uu;
  Ffm *ffm;
  DParam *p;
  Real prior_weight = 1.0 / 1.8;
  /* Compute the prior for each sentence */
  for (n = 0; n < nn; n ++){
    priors[n] = 0;  /* Log space */
    for (w = 0; w < ss[n]->ww; w ++){
      ffm = ss[n]->ws[w]->ffm[0];
      for (i = 0; i < ffm->ii; i ++){
    	p = (DParam *)(ffm->p[i]);
    	priors[n] += tt * my_log(p->kk);
      }
      /* Since the transition matrix is almost left-to-right */
      /* 	 We omit this prior */
      uu = VLENGTH(ss[n]->ws[w]->hmm->B);
      priors[n] += tt * my_log(uu);
    }
    /* Add a weight for the prior */
    priors[n] *= prior_weight;
  }
}

/* needswork: remove bm, decide the similarity just by the index 
   asumption: order of box names is the same in every frame
*/
void prune_tracker_states
(Sentence s, int nTracks, IVec *roles, int top_n, int ii, IVec *ped, IVec **ped2){

  int *js, nRoles, w, *r, *eligible, jjj, u, i;
  int eligible_tot, cnt, *ks, jj, j, k;
  IVec ed, *ed2;

  jjj = 1;
  for (i = 0; i < nTracks; i ++) jjj *= ii;
  js = safe_malloc(nTracks * sizeof *js);
  eligible = safe_malloc(jjj * sizeof *eligible);
  eligible_tot = 0;
  for (u = 0; u < jjj; u ++){
    eligible[u] = 1;
    if (model_constraint == NO_DUPLICATE_MODELS){
      decode_box_states(u, ii, nTracks, js);
      for (w = 0; w < s->ww; w ++){
	assert(s->ws[w]->n_roles == VLENGTH(roles[w]));
	nRoles = VLENGTH(roles[w]);
	assert(nRoles <= 3); 
	if (nRoles == 1) continue;
	r = VECTOR(roles[w]);
	assert(r[0] != r[1]);
	if (js[r[0]]/top_n == js[r[1]]/top_n){
	  eligible[u] = 0;
	  break;
	}
	/* when nRoles = 3, only check the similarity 1-2 and 2-3 */
	if (nRoles == 3){
	  if (js[r[1]]/top_n == js[r[2]]/top_n){
	    eligible[u] = 0;
	    break;
	  }
	}
      }
    }
    if (eligible[u] == 1) eligible_tot ++;
  }
  ed = allocIVec(eligible_tot);
  cnt = 0;
  for (u = 0; u < jjj; u ++)
    if (eligible[u] == 1){
      U(cnt) = u;
      cnt ++;
    }
  free(eligible);
  /* Now for each box state, compute the previously eligible box states such
     that the box similarity score between each eligible state and the
     current state is not 0
  */
  jj = VLENGTH(ed);
  ks = safe_malloc(nTracks * sizeof *ks);
  ed2 = safe_malloc(jj * sizeof *ed2);
  eligible = safe_malloc(jj * sizeof *eligible);
  for (j = 0; j < jj; j ++){
    decode_box_states(U(j), ii, nTracks, js);
    eligible_tot = 0;
    for (k = 0; k < jj; k ++){
      eligible[k] = 1;
      decode_box_states(U(k), ii, nTracks, ks);
      for (i = 0; i < nTracks; i ++)
	if (js[i]/top_n != ks[i]/top_n){
	  eligible[k] = 0;
	  break;
	}
      if (eligible[k] == 1)
	eligible_tot ++;
    }
    ed2[j] = allocIVec(eligible_tot);
    cnt = 0;
    for (k = 0; k < jj; k ++)
      if (eligible[k] == 1){
	J(cnt,j) = k;
	cnt ++;
      }
  }
  *ped = ed;
  *ped2 = ed2;
  free(eligible);
  free(ks);
  free(js);
}

void prune_xHMM_states(Sentence s, int uuu, IVec *pes, IVec **pes2, IVec **pes3){
  int u, v, *us, *vs, *eligible, eligible_tot, w, x, cnt, uu, vv, boundary;
  IVec es, *es2, *es3;
  Hmm **hmms;
  
  if (PAD_ALL)
    boundary = MIN_DURATION - 1;
  else 
    boundary = 0;

  hmms = safe_malloc(s->ww * sizeof *hmms);
  for (w = 0; w < s->ww; w ++)
    hmms[w] = s->ws[w]->hmm;
  
  us = safe_malloc(s->ww * sizeof *us);
  eligible = safe_malloc(uuu * sizeof *eligible);
  eligible_tot = 0;
  
  for (x = 0; x < s->ww; x ++)
    /* Todo: assume that state number is either 1 or some N (3) */
    if (s->ws[x]->uu > 1)
      break;
  /* This happens because a noun can be treated as a sentence */
  if (x == s->ww) x = 0;

  for (u = 0; u < uuu; u ++){
    eligible[u] = 1;
    if (state_constraint == TIE_STATES){
      HMM_decodeXStates(us, hmms, u, s->ww);
      for (w = 0; w < s->ww; w ++){
	if (s->ws[w]->uu > 1 && us[w] != us[x]){
	  eligible[u] = 0;
	  break;
	}
      }      
    }
    if (eligible[u] == 1) eligible_tot ++;
  }
  es = allocIVec(eligible_tot);
  cnt = 0;
  for (u = 0; u < uuu; u ++)
    if (eligible[u] == 1){
      W(cnt) = u;
      cnt ++;
    }
  free(eligible);
  /* prune the forward and backward states */
  uu = VLENGTH(es);
  vs = safe_malloc(s->ww * sizeof *vs);
  es2 = safe_malloc(uu * sizeof *es2);
  es3 = safe_malloc(uu * sizeof *es3);
  eligible = safe_malloc(uu * sizeof *eligible);
  for (u = 0; u < uu; u ++){
    HMM_decodeXStates(us, hmms, W(u), s->ww);
    // backward (v)-->u
    eligible_tot = 0;
    for (v = 0; v < uu; v ++){
      eligible[v] = 1;
      HMM_decodeXStates(vs, hmms, W(v), s->ww);
      /* Todo: may prune upper-triangular even without duration */
      if (DURATION){		
	for (w = 0; w < s->ww; w ++){
	  vv = padded_uu(hmms[w]->uu);	
	  if (vs[w] % MIN_DURATION != boundary && us[w] == vs[w] + 1)
	    continue;	
	  if ((vs[w] % MIN_DURATION == boundary || vs[w] == vv - 1)
	      && (us[w] == vs[w] || us[w] == vs[w] + 1))
	    continue;
	  eligible[v] = 0;
	  break;
	}
      }
      if (eligible[v] == 1) eligible_tot ++;   
    }
    es2[u] = allocIVec(eligible_tot);
    cnt = 0;
    for (v = 0; v < uu; v ++)
      if (eligible[v] == 1){
	V(cnt,u) = v;
	cnt ++;
      }
    // forward u-->(v)
    eligible_tot = 0;
    for (v = 0; v < uu; v ++){
      eligible[v] = 1;
      HMM_decodeXStates(vs, hmms, W(v), s->ww);
      /* Todo: may prune upper-triangular even without duration */
      if (DURATION){
	for (w = 0; w < s->ww; w ++){
	  vv = padded_uu(hmms[w]->uu);	
	  if (us[w] % MIN_DURATION != boundary && vs[w] == us[w] + 1)
	    continue;
	  if ((us[w] % MIN_DURATION == boundary || us[w] == vv - 1)
	      && (vs[w] == us[w] || vs[w] == us[w] + 1))
	    continue;
	  eligible[v] = 0;
	  break;
	}
      }
      if (eligible[v] == 1) eligible_tot ++;      
    }
    es3[u] = allocIVec(eligible_tot);
    cnt = 0;
    for (v = 0; v < uu; v ++)
      if (eligible[v] == 1){
	X(cnt,u) = v;
	cnt ++;
      }
  }
  
  *pes = es;
  *pes2 = es2;
  *pes3 = es3;

  free(us);
  free(vs);
  free(hmms);
  free(eligible);
}

void compute_crossed_likelihood(IVec ed, IVec es, Sentence s, int nTracks, IVec *roles, 
				int ***phi1, int ****phi2, int ****phi0, int tt, int ii, 
				Real **outL, Real ****xh, M0 *m0){
  int *js, *us, nRoles, w, *r, t, j, u, jj, x, vv, d1, d2, d3, _inf_idx, i;
  Hmm **hmms;
  RVec *data;
  Real *xw, *l;
  Ffm *ffm;
  Features f;
  jj = VLENGTH(ed);
  vv = VLENGTH(es);

  us = safe_malloc(s->ww * sizeof *us);
  hmms = safe_malloc(s->ww * sizeof *hmms);
  data = safe_malloc(s->ww * sizeof *data);
  xw = safe_malloc(s->ww * sizeof *xw);
  l = safe_malloc(s->ww * sizeof *l);
  js = safe_malloc(nTracks * sizeof *js);

  for (w = 0; w < s->ww; w ++){
    hmms[w] = s->ws[w]->hmm;
    assert(s->ws[w]->n_roles == VLENGTH(roles[w]));
  }
  
  for (t = 0; t < tt; t ++){
    for (j = 0; j < jj; j ++){
      decode_box_states(U(j), ii, nTracks, js);
      for (w = 0; w < s->ww; w ++){
	/* For model w in the sentence s, pick the tracks according to its roles */
	nRoles = VLENGTH(roles[w]);
	/* For now, each model has no more than three roles */
	assert(nRoles <= 3 && nRoles >= 1);  
	r = VECTOR(roles[w]);
	d1 = js[r[0]];	
	m0->d0[0] = m0->p0[r[0]];
	d2 = -1;
	if (nRoles >= 2){
	  assert(r[0] != r[1]);
	  d2 = js[r[1]];
	  m0->d0[1] = m0->p0[r[1]];
	}
	d3 = -1;
	if (nRoles >= 3){
	  assert(r[1] != r[2] && r[0] != r[2]);
	  d3 = js[r[2]];
	  m0->d0[2] = m0->p0[r[2]];
	}
	/* Compute a feature vector for this model */
	f = new_feature(s->ws[w]->pos, d1, d2, d3, phi1[t], phi2[t], phi0?phi0[t]:NULL, m0);
	data[w] = allocRVec(f->ii);
	memcpy(VECTOR(data[w]), f->v[0], f->ii * sizeof(Real));
	freeFeatures(f);
      }
      for (u = 0; u < vv; u ++){
	HMM_decodeXStates(us, hmms, W(u), s->ww);
	x = u * jj + j;
	outL[t][x] = 0;
	_inf_idx = 0;
	for (w = 0; w < s->ww; w ++){
	  l[w] = FFM_likelihood_one
	    (OUTPUT_MODEL(s, w, effective_state(us[w])), data[w], xh?xh[t][x][w]:NULL);
	  if (l[w] == NEGATIVE_INFINITY)
	    _inf_idx |= (1 << w);
	  else
	    outL[t][x] += l[w];
	  if (xh) xw[w] = NEGATIVE_INFINITY;
	}
	if (xh){
	  for (w = 0; w < s->ww; w ++){
	    if ((_inf_idx & ~(1 << w)) == 0){ /* if no other l[w'] = -inf */
	      xw[w] = outL[t][x];
	      if (!_inf_idx) xw[w] -= l[w];
	    }
	    ffm = OUTPUT_MODEL(s, w, effective_state(us[w]));
	    for (i = 0; i < ffm->ii; i ++)
	      xh[t][x][w][i] += xw[w];
	  }
	}
	if (_inf_idx)
	  outL[t][x] = NEGATIVE_INFINITY;	
      }
      for (w = 0; w < s->ww; w ++)
	freeRVec(data[w]);
    }
  }
  free(us);
  free(hmms);
  free(data);
  free(xw);
  free(l);
  free(js);
  return;
}

/* Similar to viterbi tracker The difference is the MAX replaced with SUM
*/
Real viterbi_tracker_score(Real ***bps, Real **bss, int nTracks, int ii, int tt, IVec ed, IVec *ed2){
  int *js = safe_malloc(nTracks * sizeof *js);
  int *ks = safe_malloc(nTracks * sizeof *ks);
  Real **alpha, overall_score;
  Real pair, single;
  int i, t, idx = 0, jj, kk, j, k;

  jj = VLENGTH(ed);
  alpha = safe_malloc(jj * sizeof *alpha); /* log space */;
  /* t = 0 */
  for (j = 0; j < jj; j ++){
    alpha[j] = safe_malloc(2 * sizeof *(alpha[j]));
    decode_box_states(U(j), ii, nTracks, js);
    alpha[j][0] = 0;
    for (i = 0; i < nTracks; i ++)
      alpha[j][0] += bss[0][js[i]];
  }
  /* t >= 1 */
  for (t = 1; t < tt; t ++){
    idx = t % 2;
    for (j = 0; j < jj; j ++){
      decode_box_states(U(j), ii, nTracks, js);
      alpha[j][idx] = NEGATIVE_INFINITY;
      kk = VLENGTH(ed2[j]);
      for (k = 0; k < kk; k ++){
	decode_box_states(U(J(k,j)), ii, nTracks, ks);
	pair = 0;
	for (i = 0; i < nTracks; i ++)
	  pair += bps[t-1][ks[i]][js[i]];
	assert(pair != NEGATIVE_INFINITY);
	alpha[j][idx] = add_exp(alpha[j][idx], alpha[J(k,j)][1-idx] + pair);
      }
      single = 0;
      for (i = 0; i < nTracks; i ++)
	single += bss[t][js[i]];
      alpha[j][idx] += single;
    }
  }
  overall_score = NEGATIVE_INFINITY;
  for (j = 0; j < jj; j ++)
    overall_score = add_exp(overall_score, alpha[j][idx]);
  /* Clean up */
  free(js);
  free(ks);
  for (j = 0; j < jj; j ++)
    free(alpha[j]);
  free(alpha);
  return overall_score;
}

void compute_sentence_alpha(IVec ed, IVec *ed2, IVec es, IVec *es2, int *final, Sentence s, int ii, int tt, int nTracks,
			    Real **a, Real *b, Real ***bps, Real **bss, Real **outL, Real ***xa, Real **xb, Real **alpha, 
			    Real **alpha_p, Real ****xi, M0 *m0){
  int *js, *ks, u, v, i, t, jj, x, y, j, k, kk, vv, zz, phi, chi, j0 = m0->j0;
  int w, ww=s->ww, *vs, *us, ev, boundary, _uu;
  Real single, pair;
  Hmm **hmms;

  if (PAD_ALL)
    boundary = MIN_DURATION - 1;
  else 
    boundary = 0;

  js = safe_malloc(nTracks * sizeof *js);
  ks = safe_malloc(nTracks * sizeof *ks);
  vs = safe_malloc(ww * sizeof *vs);
  us = safe_malloc(ww * sizeof *us);
  hmms = safe_malloc(ww * sizeof *hmms);

  for (w = 0; w < ww; w ++)
    hmms[w] = s->ws[w]->hmm;
  jj = VLENGTH(ed);
  vv = VLENGTH(es);
  /* initialize */
  for (t = 0; t < tt; t ++)
    for (x = 0; x < jj*vv; x ++){
      alpha[t][x] = NEGATIVE_INFINITY;
      if (alpha_p){
	alpha_p[t][x] = NEGATIVE_INFINITY;	
	for (w = 0; w < ww; w ++)
	  for (u = 0; u < s->ws[w]->uu; u ++)
	    xi[t][x][w][u] = NEGATIVE_INFINITY;
      }
    }
  /* t = 0 */
  for (j = (DISPLACEMENT? j0: 0); j < (DISPLACEMENT? j0+1: jj); j ++){
    decode_box_states(U(j), ii, nTracks, js);
    /* Box single score at frame 0 */
    single = 0;
    for (i = 0; i < nTracks; i ++)
      single += bss[0][js[i]];
    for (u = 0; u < vv; u ++){
      HMM_decodeXStates(us, hmms, W(u), ww);
      x = u * jj + j;      
      /* Initial state probability times output probability */
      if (alpha_p){
	alpha_p[0][x] = b[W(u)];
	alpha[0][x] = alpha_p[0][x] + outL[0][x] + single;
	for (w = 0; w < ww; w ++)
	  if (!DURATION || us[w] == 0)
	    xi[0][x][w][0] = xb[W(u)][w];
      }
      else
	alpha[0][x] = b[W(u)] + outL[0][x] + single;      
    }
  }
  kk = (DISPLACEMENT? VLENGTH(ed2[j0]): jj);
  /* t >= 1 */
  for (t = 1; t < tt; t ++){
    /* frame t */
    for (j = 0; j < kk; j ++){
      phi = (DISPLACEMENT? J(j,j0): j);
      decode_box_states(U(phi), ii, nTracks, js);
      single = 0;
      for (i = 0; i < nTracks; i ++)
	single += bss[t][js[i]];
      chi = (DISPLACEMENT? j0: j);
      for (u = 0; u < vv; u ++){
	x = u * jj + phi;
	zz = VLENGTH(es2[u]);
	HMM_decodeXStates(us, hmms, W(u), ww);
	if (t < tt-1 || final[W(u)]){
	  /* frame t - 1 */
	  for (k = 0; k < VLENGTH(ed2[chi]); k ++){
	    decode_box_states(U(J(k,chi)), ii, nTracks, ks);
	    pair = 0;
	    for (i = 0; i < nTracks; i ++)
	      pair += bps[t-1][ks[i]][js[i]];
	    assert(pair != NEGATIVE_INFINITY);
	    for (v = 0; v < zz; v ++){
	      y = V(v,u) * jj + J(k,chi);
	      if (alpha_p){
		alpha_p[t][x] = add_exp(alpha_p[t][x], alpha[t-1][y] + pair + a[W(V(v,u))][W(u)]);
		/* Todo: This decoding trick is not efficient; should optimize */
		HMM_decodeXStates(vs, hmms, W(V(v,u)), ww);
		for (w = 0; w < ww; w ++){
		  ev = effective_state(vs[w]);
		  _uu = padded_uu(hmms[w]->uu);
		  if (!DURATION ||
		      ((vs[w] % MIN_DURATION == boundary || vs[w] == _uu - 1)
		       && (vs[w] == us[w] || us[w] == vs[w] + 1)))
		    xi[t][x][w][ev] = add_exp(xi[t][x][w][ev],
					      alpha[t-1][y]
					      + xa[W(V(v,u))][W(u)][w]
					      + pair);
		}
	      }
	      else
		alpha[t][x] = add_exp(alpha[t][x], alpha[t-1][y] + pair + a[W(V(v,u))][W(u)]);
	    }
	  } 
	  if (alpha_p)
	    alpha[t][x] = alpha_p[t][x] + single + outL[t][x];
	  else
	    alpha[t][x] += single + outL[t][x];
	}
      }
    }
  }
  free(js);
  free(ks);
  free(vs);
  free(us);
  free(hmms);
}

void compute_sentence_beta(IVec ed, IVec *ed2, IVec es, IVec *es3, int *final, int ii, int tt, int nTracks, 
			   Real **a, Real ***bps, Real **bss, Real **outL, Real **beta, M0 *m0){
  int *js, *ks, u, v, i, t, jj, x, y, j, k, kk, vv, zz, phi, chi, j0 = m0->j0;
  Real single, pair;
  jj = VLENGTH(ed);
  vv = VLENGTH(es);
  js = safe_malloc(nTracks * sizeof *js);
  ks = safe_malloc(nTracks * sizeof *ks);
  /* initialize */
  for (t = 0; t < tt; t ++)
    for (x = 0; x < jj*vv; x ++)
      beta[t][x] = NEGATIVE_INFINITY;

  kk = (DISPLACEMENT? VLENGTH(ed2[j0]): jj);
  /* t = tt - 1 */
  for (j = 0; j < kk; j ++)
    for (u = 0; u < vv; u ++){
      phi = (DISPLACEMENT? J(j,j0): j);
      x = u * jj + phi;
      if (final[W(u)])
	beta[tt-1][x] = 0;
    }
  /* t <= tt - 2 */
  for (t = tt - 2; t >= 0; t --){
    /* frame t */
    for (j = 0; j < kk; j ++){
      phi = (DISPLACEMENT? J(j,j0): j);
      decode_box_states(U(phi), ii, nTracks, js);
      chi = (DISPLACEMENT? j0: j);
      /* exclude t == 0 && phi != j0 for DISPLACEMENT */
      if (!(DISPLACEMENT && t == 0 && phi != j0)){ 
	for (u = 0; u < vv; u ++){
	  x = u * jj + phi;
	  zz = VLENGTH(es3[u]);
	  /* frame t + 1 */	  
	  for (k = 0; k < VLENGTH(ed2[chi]); k ++){
	    decode_box_states(U(J(k,chi)), ii, nTracks, ks);
	    single = pair = 0;
	    for (i = 0; i < nTracks; i ++){
	      pair += bps[t][js[i]][ks[i]];
	      single += bss[t+1][ks[i]];
	    }
	    assert(pair != NEGATIVE_INFINITY);
	    for (v = 0; v < zz; v ++){
	      y = X(v,u) * jj + J(k,chi);
	      beta[t][x] = add_exp(beta[t][x], beta[t+1][y] + pair 
				   + a[W(u)][W(X(v,u))] + outL[t+1][y] + single);
	    }
	  }
	}
      }
    }
  }
  free(js);
  free(ks);
}

/* gamma is in linear space */
void compute_sentence_gamma
(int uu, int tt, Real **alpha, Real **beta, Real **gamma){
  int t, u;
  for (t = 0; t < tt; t ++)
    for (u = 0; u < uu; u ++)
      gamma[t][u] = alpha[t][u] + beta[t][u];
}

Real compute_sentence_statistics(Sentence s, Hmm *xhmm, int nTracks, IVec *roles,
				 Real **bss, Real ***bps, int ***phi1, int ****phi2, 
				 int ****phi0, int tt, int ii, int final_state,
				 IVec ed, IVec *ed2, IVec es, IVec *es2, IVec *es3,
				 /* Ouput */
				 Real **gamma, Real **alpha, Real **beta, Real **outL, M0 *m0){
 
  int vv, jj, uu, u, w, *us, *final, *js, i;
  Real **a, *b, likelihood;
  Hmm **hmms;

  js = safe_malloc(nTracks * sizeof *js);

  a = MATRIX(xhmm->A);
  b = VECTOR(xhmm->B);
  jj = VLENGTH(ed);
  vv = VLENGTH(es);
  uu = jj * vv;

  hmms = safe_malloc(s->ww * sizeof *hmms);
  for (w = 0; w < s->ww; w ++)
    hmms[w] = s->ws[w]->hmm;
  us = safe_malloc(s->ww * sizeof *us);
  final = safe_malloc(xhmm->uu * sizeof *final);
  
  /* compute eligible final state
     this is computed for states before pruning
   */
  for (u = 0; u < xhmm->uu; u ++){
    final[u] = TRUE;
    if (final_state){
      HMM_decodeXStates(us, hmms, u, s->ww);
      for (w = 0; w < s->ww; w ++)
	/* if the state is not at the final, ignore */
	if (us[w] != padded_uu(hmms[w]->uu)-1)
	  break;
      if (w < s->ww) 
	final[u] = FALSE;
    }
  }

  if (DISPLACEMENT){
    /* select the participant->track mapping for the first frame */
    assert(nTracks <= MAX_NTRACKS);
    memset(m0->p0, -1, MAX_NTRACKS * sizeof *(m0->p0));
    decode_box_states(U(m0->j0), ii, nTracks, js);
    for (i = 0; i < nTracks; i ++)
      m0->p0[i] = js[i];
  }

  /* Compute crossed likelihood */
  compute_crossed_likelihood(ed, es, s, nTracks, roles, phi1, phi2, phi0, tt, ii, outL, NULL, m0);
  /* Calculate gamma and sentence likelihood */
  compute_sentence_alpha(ed, ed2, es, es2, final, s, ii, tt, nTracks, a, b, bps, bss, outL, NULL, NULL, alpha, NULL, NULL, m0);

  compute_sentence_beta(ed, ed2, es, es3, final, ii, tt, nTracks, a, bps, bss, outL, beta, m0);

  likelihood = NEGATIVE_INFINITY;
  for (u = 0; u < uu; u ++)
    likelihood = add_exp(likelihood, alpha[tt-1][u]);

  compute_sentence_gamma(uu, tt, alpha, beta, gamma);

  free(hmms);
  free(js);
  free(us);
  free(final);
  return likelihood;
}

void HMM_updateXModel(Sentence s, Sentence tmp, Hmm *xhmm, int tt, int ii, Real **bss, Real ***bps, 
		      IVec ed, IVec *ed2, IVec es, IVec *es3, /* forward */
		      int nTracks, Real **alpha, Real **beta, Real **outL, Real *gamma0, M0 *m0){
  int t, u, v, i, w, ww=s->ww, *js, *ks, *us, *vs, eu, ev;
  int kk, jj, j, k, x, y, vv, zz, uu, boundary, phi, chi, j0 = m0->j0;
  Real **a, **na, *nb, psi, single, pair;
  Hmm **hmms;
  a = MATRIX(xhmm->A);
  jj = VLENGTH(ed);
  vv = VLENGTH(es);
  assert(ww == tmp->ww);
  
  if (PAD_ALL)
    boundary = MIN_DURATION - 1;
  else 
    boundary = 0;

  js = safe_malloc(nTracks * sizeof *js);
  ks = safe_malloc(nTracks * sizeof *ks);
  vs = safe_malloc(ww * sizeof *vs);
  us = safe_malloc(ww * sizeof *us);
  hmms = safe_malloc(ww * sizeof *hmms);

  for (w = 0; w < ww; w ++)
    hmms[w] = s->ws[w]->hmm;
  
  kk = (DISPLACEMENT? VLENGTH(ed2[j0]): jj);
  for (t = 0; t < tt - 1; t ++){
    /* frame t */
    for (j = 0; j < kk; j ++){
      phi = (DISPLACEMENT? J(j,j0): j);
      decode_box_states(U(phi), ii, nTracks, js);
      chi = (DISPLACEMENT? j0: j);
      for (u = 0; u < vv; u ++){
	HMM_decodeXStates(us, hmms, W(u), ww);
	x = u * jj + phi;
	zz = VLENGTH(es3[u]);
	/* Update initial probability at time 0 */
	if (t == 0){
	  for (w = 0; w < ww; w ++){
	    nb = VECTOR(SENTENCE_HMM_B(tmp, w));	    
	    /* only update the initial states */
	    if (!DURATION || us[w] == 0){
	      eu = effective_state(us[w]);
	      nb[eu] = add_exp(nb[eu], gamma0[x]);
	    }
	  }
	}
	/* frame t + 1 */
	for (k = 0; k < VLENGTH(ed2[chi]); k ++){
	  decode_box_states(U(J(k,chi)), ii, nTracks, ks);
	  pair = single = 0;
	  for (i = 0; i < nTracks; i ++){
	    pair += bps[t][js[i]][ks[i]];
	    single += bss[t+1][ks[i]];
	  }
	  assert(pair != NEGATIVE_INFINITY);
	  for (v = 0; v < zz; v ++){
	    HMM_decodeXStates(vs, hmms, W(X(v,u)), ww);
	    y = X(v,u) * jj + J(k,chi);
	    psi = alpha[t][x] + (a[W(u)][W(X(v,u))] + pair)
	      + (outL[t+1][y] + single) + beta[t+1][y];
	    for (w = 0; w < ww; w ++){	      
	      uu = padded_uu(hmms[w]->uu);
	      na = MATRIX(SENTENCE_HMM_A(tmp, w));
	      assert(us[w] >= 0 && us[w] < uu);
	      assert(vs[w] >= 0 && vs[w] < uu);	      
	      /* only update the transition between states that are not padded */
	      if (!DURATION || 
		  ((us[w] % MIN_DURATION == boundary || us[w] == uu - 1)
		   && (vs[w] == us[w] || vs[w] == us[w] + 1))){
		eu = effective_state(us[w]);
		ev = effective_state(vs[w]);
		na[eu][ev] = add_exp(na[eu][ev], psi);
	      }
	    }
	  }
	}
      }
    }
  }
  free(js);
  free(ks);
  free(vs);
  free(us);
  free(hmms);
}

void FFM_updateXModel(Sentence s, Sentence tmp, IVec *roles, Real **gamma, IVec ed, IVec *ed2,
		      IVec es, int nTracks, int ***phi1, int ****phi2, int ****phi0,
		      int tt, int ii, M0 *m0){
  int t, i, u, j, jj, *js, w, ww=s->ww, vv, *us, nRoles, *r, x, z, v, kk, phi;
  int d1, d2, d3, j0 = m0->j0;
  Hmm **hmms;
  Features *fs;
  DParam *np;
  jj = VLENGTH(ed);
  vv = VLENGTH(es);
  assert(ww == tmp->ww);

  js = safe_malloc(nTracks * sizeof *js);
  us = safe_malloc(ww * sizeof *us);
  hmms = safe_malloc(ww * sizeof *hmms);
  fs = safe_malloc(ww * sizeof *fs);

  for (w = 0; w < ww; w ++){
    hmms[w] = s->ws[w]->hmm;
    assert(s->ws[w]->n_roles == VLENGTH(roles[w]));
  }
  
  kk = (DISPLACEMENT? VLENGTH(ed2[j0]): jj);
  for (t = 0; t < tt; t ++){
    for (j = 0; j < kk; j ++){
      phi = (DISPLACEMENT? J(j,j0): j);
      decode_box_states(U(phi), ii, nTracks, js);
      for (w = 0; w < ww; w ++){
	nRoles = VLENGTH(roles[w]);
	assert(nRoles <= 3 && nRoles >= 1);
	r = VECTOR(roles[w]);
	d1 = js[r[0]];
	m0->d0[0] = m0->p0[r[0]];
	d2 = -1;
	d3 = -1;
	if (nRoles >= 2){
	  assert(r[0] != r[1]);
	  d2 = js[r[1]];
	  m0->d0[1] = m0->p0[r[1]];
	}
	if (nRoles >= 3){
	  assert(r[1] != r[2] && r[0] != r[2]);
	  d3 = js[r[2]];
	  m0->d0[2] = m0->p0[r[2]];
	}
	/* Compute the feature for this model at time t 
	 */
	fs[w] = new_feature(s->ws[w]->pos, d1, d2, d3, phi1[t], phi2[t], phi0[t], m0);
	assert(fs[w]->ii == s->ws[w]->ffm[0]->ii);
      }
      for (u = 0; u < vv; u ++){
	HMM_decodeXStates(us, hmms, W(u), ww);
	x = u * jj + phi;
	for (w = 0; w < ww; w ++){
	  v = effective_state(us[w]);
	  for (i = 0; i < fs[w]->ii; i ++){	    
	    assert(s->ws[w]->ffm[v]->ft[i] == FT_DISCRETE);
	    np = SENTENCE_FFM_P(tmp, w, v, i);
	    z = (int)(fs[w]->v[0][i]);
	    assert(z >= 0 && z < np->kk);
	    np->p[z] = add_exp(np->p[z], gamma[t][x]);
	  }
	}
      }
      for (w = 0; w < ww; w ++)
      	freeFeatures(fs[w]);
    }
  }
  free(js);
  free(us);
  free(hmms);
  free(fs);
}

Real viterbi_sentence_tracker
(Sentence s, int nTracks, IVec *roles, int ***phi1, int ****phi2, int ****phi0, Real **bss, 
 Real ***bps, int tt, int ii, int top_n, int sum_or_max, int final_state, int *box_sequence){
  
  int t, j, jj, k, kk, u, vv, uu, zz, v, w, x, y, j0;
  int *js, *ks, current_state=-1, i, *us, phi, chi, best_state=-1, **path, **best_path;  
  Real **alpha, prior, track_score, return_value=0, transition_score, single, pair, sentence_score, **a, *b, **outL;
  Hmm **hmms, *xhmm;
  IVec ed, *ed2, es, *es2, *es3;
  M0 *m0 = allocM0();

  us = safe_malloc(s->ww * sizeof *us);
  hmms = safe_malloc(s->ww * sizeof *hmms);
  for (w = 0; w < s->ww; w ++)
    hmms[w] = s->ws[w]->hmm;
  xhmm = initializeXHMM(hmms, s->ww, NULL, NULL);

  prune_tracker_states(s, nTracks, roles, top_n, ii, &ed, &ed2);
  /* left-to-right; sparse representation; state tying */
  prune_xHMM_states(s, xhmm->uu, &es, &es2, &es3);

  jj = VLENGTH(ed);  
  vv = VLENGTH(es);
  uu = jj * vv;

  alpha = safe_malloc(tt * sizeof *alpha); 
  outL = safe_malloc(tt * sizeof *outL);
  path = safe_malloc(tt * sizeof *path);
  best_path = safe_malloc(tt * sizeof *best_path);  
  for (t = 0; t < tt; t ++){
    alpha[t] = safe_malloc(uu * sizeof *(alpha[t]));
    outL[t] = safe_malloc(uu * sizeof *(outL[t]));
    path[t] = safe_malloc(uu * sizeof *(path[t]));
    best_path[t] = safe_malloc(uu * sizeof *(best_path[t]));
  }
  js = safe_malloc(nTracks * sizeof *js);
  ks = safe_malloc(nTracks * sizeof *ks);
   
  a = MATRIX(xhmm->A);
  b = VECTOR(xhmm->B);
  assert(nTracks <= MAX_NTRACKS);
  /* either likelihood or MAP */
  sentence_score = NEGATIVE_INFINITY;

  for (j0 = 0; j0 < (DISPLACEMENT?jj:1); j0 ++){    
    m0->j0 = j0;
    if (DISPLACEMENT){
      decode_box_states(U(j0), ii, nTracks, js);
      memset(m0->p0, -1, MAX_NTRACKS * sizeof *(m0->p0));
      /* get the initial participant-to-detection mapping */
      for (i = 0; i < nTracks; i ++)
	m0->p0[i] = js[i];
    }
    /* re-compute the likelihood given the new anchor detections p0 */
    compute_crossed_likelihood(ed, es, s, nTracks, roles, phi1, phi2, phi0, tt, ii, outL, NULL, m0);

    for (t = 0; t < tt; t ++)
      for (x = 0; x < uu; x ++)
	alpha[t][x] = NEGATIVE_INFINITY;
    /* t = 0 */
    for (j = (DISPLACEMENT? j0: 0); j < (DISPLACEMENT? j0+1: jj); j ++){
      decode_box_states(U(j), ii, nTracks, js);
      single = 0;
      /* Box single score at frame 0 */
      for (i = 0; i < nTracks; i ++)
	single += bss[0][js[i]];
      for (u = 0; u < vv; u ++){
	x = u * jj + j;
	alpha[0][x] = b[W(u)] + outL[0][x] + single;
	path[0][x] = -1;
      }
    }
    kk = (DISPLACEMENT? VLENGTH(ed2[j0]): jj);
    /* t >= 1 */
    for (t = 1; t < tt; t ++){
      // -------- tracker lattice (t) --------- //
      for (j = 0; j < kk; j ++){
	phi = (DISPLACEMENT? J(j,j0): j);
  	decode_box_states(U(phi), ii, nTracks, js);
  	single = 0;
  	for (i = 0; i < nTracks; i ++)
  	  single += bss[t][js[i]];
	chi = (DISPLACEMENT? j0: j);
  	// --------- word lattice (t) --------- //
  	for (u = 0; u < vv; u ++){
  	  x = u * jj + phi;
  	  zz = VLENGTH(es2[u]);
  	  // ----------- tracker lattice (t-1) --------- //
	  for (k = 0; k < VLENGTH(ed2[chi]); k ++){
  	    decode_box_states(U(J(k,chi)), ii, nTracks, ks);
  	    pair = 0;
  	    for (i = 0; i < nTracks; i ++)
  	      pair += bps[t-1][ks[i]][js[i]];
  	    assert(pair != NEGATIVE_INFINITY);
  	    // ------------ word lattice (t-1) ----------- //
  	    for (v = 0; v < zz; v ++){
  	      y = V(v,u) * jj + J(k,chi);
  	      transition_score = alpha[t-1][y] + pair + a[W(V(v,u))][W(u)];
  	      if (sum_or_max == 0)
  		alpha[t][x] = add_exp(alpha[t][x], transition_score);
  	      else{  /* track the state sequence */
  		if (alpha[t][x] < transition_score){
  		  alpha[t][x] = transition_score;
  		  path[t][x] = y;
  		}
  	      }
  	    }
  	  }
  	  alpha[t][x] += single + outL[t][x];
  	}
      }
    }
    /* compute likelihood */
    if (sum_or_max == 0){
      return_value = NEGATIVE_INFINITY;
      for (u = 0; u < vv; u ++){
  	HMM_decodeXStates(us, hmms, W(u), s->ww);
  	if (final_state){
  	  for (w = 0; w < s->ww; w ++)
  	    /* if the state is not at the final, ignore */
  	    if (us[w] != padded_uu(hmms[w]->uu)-1)
  	      break;
  	  if (w < s->ww) continue;
  	}
  	for (j = 0; j < jj; j ++){
  	  x = u * jj + j;
  	  return_value = add_exp(return_value, alpha[tt - 1][x]);
  	}
      }
      sentence_score = add_exp(sentence_score, return_value);
    }
    /* compute most possible state sequence */
    else{
      return_value = NEGATIVE_INFINITY;
      for (u = 0; u < vv; u ++){
  	HMM_decodeXStates(us, hmms, W(u), s->ww);
  	if (final_state){
  	  for (w = 0; w < s->ww; w ++)
  	    /* if the state is not at the final, ignore */
  	    if (us[w] != padded_uu(hmms[w]->uu)-1)
  	      break;
  	  if (w < s->ww) continue;
  	}
  	for (j = 0; j < jj; j ++){
  	  x = u * jj + j;
  	  if (return_value < alpha[tt-1][x]){
  	    return_value = alpha[tt-1][x];
  	    current_state = x;
  	  }
  	}
      }
      if (return_value > sentence_score){
  	sentence_score = return_value;
  	best_state = current_state;
  	/* copy the path */
  	for (t = 0; t < tt; t ++)
  	  for (u = 0; u < uu; u ++)
  	    best_path[t][u] = path[t][u];
      }
    }
  }
  if (sum_or_max == 1 && sentence_score != NEGATIVE_INFINITY){
    /* Get the box sequence */
    for (t = tt - 1; t >= 0; t --){
      box_sequence[t] = U(best_state % jj);
      best_state = best_path[t][best_state];
    }
    assert(best_state == -1);
  }
  /* No track eligible */
  if (sentence_score == NEGATIVE_INFINITY){
    printf("viterbi-sentence-track-in-c, all costs were infinity\n");
  }
  track_score = viterbi_tracker_score(bps, bss, nTracks, ii, tt, ed, ed2);
  sentence_score -= track_score;
  if (NO_PRIOR == FALSE){
    compute_sentence_priors(&s, 1, tt, &prior);
    sentence_score += prior;
  }
  /* Clean up */
  for (t = 0; t < tt; t ++){
    free(alpha[t]);
    free(outL[t]);
    free(path[t]);
    free(best_path[t]);
  }
  for (j = 0; j < VLENGTH(ed); j ++)
    freeIVec(ed2[j]);
  for (u = 0; u < VLENGTH(es); u ++){
    freeIVec(es2[u]);
    freeIVec(es3[u]);
  }
  free(hmms);
  free(us);
  free(path);
  free(alpha);
  free(best_path);
  free(ks);
  free(js);
  free(outL);
  freeIVec(ed);
  freeIVec(es);
  free(ed2);
  free(es2);
  free(es3);
  freeHMM(xhmm);
  free(m0);
  return sentence_score;
}

Real sentence_maximum_one
(Sentence s, int nTracks, IVec *roles, int ***phi1, int ****phi2, int ****phi0, 
 Real **bss, Real ***bps, int tt, int ii, int top_n, int final_state, int *box_sequence){
  return viterbi_sentence_tracker
    (s, nTracks, roles, phi1, phi2, phi0, bss, bps, tt, ii, top_n, 1, final_state, box_sequence);
}

Real *sentence_likelihoods_one_video
(Sentence *ss, int nn, int *nTracks, IVec **roles, int ***phi1, int ****phi2, int ****phi0, 
 Real **bss, Real ***bps, int tt, int ii, int top_n, int final_state){

  Real *likelihoods;
  int n;

  likelihoods = safe_malloc(nn * sizeof *likelihoods);

#pragma omp parallel for private(n)
  for (n = 0; n < nn; n ++)
    likelihoods[n] = viterbi_sentence_tracker
      (ss[n], nTracks[n], roles[n], phi1, phi2, phi0, bss, bps, tt, ii, top_n, 0, final_state, NULL);
  return likelihoods;
}

Real updateX(Sentence s, Sentence scratch, int nTracks, IVec *roles,
	     int final_state, int ***phi1, int ****phi2, int ****phi0,
	     Real **bss, Real ***bps, int tt, int ii, int top_n){
  IVec ed, *ed2, es, *es2, *es3;
  Real **gamma, **alpha, **beta, **outL, likelihood, sentence_likelihood;
  Real **sa, *sb, **ta, *tb;
  Hmm *xhmm, **hmms;
  Ffm *sffm, *tffm;
  int w, j, t, uu, jj, vv, u, v, i, k, j0;
  Sentence tmp;
  Model **words;
  M0 *m0 = allocM0();

  words = safe_malloc(s->ww * sizeof *words);
  for (w = 0; w < s->ww; w ++){
    words[w] = cloneModel(scratch->ws[w]);
    zeroModel(words[w]);
    linear2logModel(words[w]);
  }
  tmp = initializeSentence(s->ww, words);

  /* Allocate and initialize an xHmm for the sentence */  
  hmms = safe_malloc(s->ww * sizeof *hmms);
  for (w = 0; w < s->ww; w ++)
    hmms[w] = s->ws[w]->hmm;
  xhmm = initializeXHMM(hmms, s->ww, NULL, NULL);
 
  prune_tracker_states(s, nTracks, roles, top_n, ii, &ed, &ed2);
  prune_xHMM_states(s, xhmm->uu, &es, &es2, &es3);

  gamma = safe_malloc(tt * sizeof *gamma);
  alpha = safe_malloc(tt * sizeof *alpha);
  beta = safe_malloc(tt * sizeof *beta);
  outL = safe_malloc(tt * sizeof *outL);
  jj = VLENGTH(ed);
  vv = VLENGTH(es);
  uu = jj * vv;
  for (t = 0; t < tt; t ++){
    gamma[t] = safe_malloc(uu * sizeof *(gamma[t]));
    alpha[t] = safe_malloc(uu * sizeof *(alpha[t]));
    beta[t] = safe_malloc(uu * sizeof *(beta[t]));
    outL[t] = safe_malloc(uu * sizeof *(outL[t]));
  }
  
  sentence_likelihood = NEGATIVE_INFINITY;  

  for (j0 = 0; j0 < (DISPLACEMENT?jj:1); j0 ++){
    /* Do some preparation works for estimation
     Including gamma, alpha, beta, and posteriors
     Also return the current sentence likelihood
   */
    m0->j0 = j0;
    likelihood = compute_sentence_statistics(s, xhmm, nTracks, roles, bss, bps,
					     phi1, phi2, phi0, tt, ii, final_state, ed, ed2, 
					     es, es2, es3, gamma, alpha, beta, outL, m0);
    HMM_updateXModel(s, tmp, xhmm, tt, ii, bss, bps, ed, ed2, es, es3, nTracks, alpha, beta, outL, gamma[0], m0);

    FFM_updateXModel(s, tmp, roles, gamma, ed, ed2, es, nTracks, phi1, phi2, phi0, tt, ii, m0);
    sentence_likelihood = add_exp(sentence_likelihood, likelihood);
  }

  /* update the scratch models */
  if (sentence_likelihood != NEGATIVE_INFINITY){
    for (w = 0; w < scratch->ww; w ++){
      sa = MATRIX(scratch->ws[w]->hmm->A);
      sb = VECTOR(scratch->ws[w]->hmm->B);
      ta = MATRIX(tmp->ws[w]->hmm->A);
      tb = VECTOR(tmp->ws[w]->hmm->B);
      uu = tmp->ws[w]->uu;
      for (u = 0; u < uu; u ++){
	sb[u] += my_exp(tb[u] - sentence_likelihood);
	for (v = 0; v < uu; v ++)
	  sa[u][v] += my_exp(ta[u][v] - sentence_likelihood);
	sffm = scratch->ws[w]->ffm[u];
	tffm = tmp->ws[w]->ffm[u];
	for (i = 0; i < tffm->ii; i ++)
	  for (k = 0; k < ((DParam *)(tffm->p[i]))->kk; k ++)
	    ((DParam *)(sffm->p[i]))->p[k]
	      += my_exp(((DParam *)(tffm->p[i]))->p[k] - sentence_likelihood);
      }	
    }
  }
  /* Clean up */
  for (j = 0; j < VLENGTH(ed); j ++)
    freeIVec(ed2[j]);
  for (u = 0; u < VLENGTH(es); u ++){
    freeIVec(es2[u]);
    freeIVec(es3[u]);
  }
  for (t = 0; t < tt; t ++){
    free(gamma[t]);
    free(alpha[t]);
    free(beta[t]);
    free(outL[t]);
  }
  for (w = 0; w < s->ww; w ++)
    freeModel(words[w]);

  freeSentence(tmp);
  free(words);
  freeHMM(xhmm);
  free(hmms);
  freeIVec(ed);
  freeIVec(es);
  free(ed2);
  free(es2);
  free(es3);
  free(gamma);
  free(alpha);
  free(beta);
  free(outL);
  free(m0);
  return sentence_likelihood;
}

Real *sentence_derivatives_one_video
(Sentence *ss, Sentence *tmps, int nn, int *nTracks, IVec *roles, int ***phi1, int ****phi2, 
 int ****phi0, Real **bss, Real ***bps, int tt, int ii, int top_n, int final_state){
  
  Real *likelihoods = safe_malloc(nn * sizeof *likelihoods);
  int n, offsets[nn];

  offsets[0] = 0;
  for (n = 1; n < nn; n ++)
    offsets[n] = offsets[n-1] + ss[n-1]->ww;

#pragma omp parallel for private(n) num_threads(6)
  for (n = 0; n < nn; n ++){
    likelihoods[n] = 
      sentence_derivatives_reverse_mode(ss[n], tmps[n], nTracks[n], roles+offsets[n], phi1, phi2,
					phi0, bss, bps, tt, ii, top_n, final_state);
  }
  return likelihoods;
}

Real sentence_derivatives_reverse_mode
(Sentence s, Sentence tmp, int nTracks, IVec *roles, int ***phi1, int ****phi2, int ****phi0, Real **bss, 
 Real ***bps, int tt, int ii, int top_n, int final_state){
  
  int t, j, jj, kk, u, vv, uu, z, v, w, x, ww = s->ww, eu, ev;
  int *js, i, *us, phi, j0;  
  int d1, d2, d3, nRoles, *r, *final;
  Features *fs;
  Real return_value=0, single, sentence_score, **a, *b, **na, *nb;
  Real **alpha, **beta, **outL, ****xh, **alpha_p, ****xi, ***xa, **xb;
  DParam *np;
  Hmm **hmms, *xhmm;
  IVec ed, *ed2, es, *es2, *es3;
  M0 *m0 = allocM0();
  assert(nTracks <= MAX_NTRACKS);

  hmms = safe_malloc(ww * sizeof *hmms);
  for (w = 0; w < ww; w ++){
    hmms[w] = s->ws[w]->hmm;
    zeroModel(tmp->ws[w]);
    linear2logModel(tmp->ws[w]);
  }
  xhmm = initializeXHMM(hmms, ww, &xa, &xb);

  js = safe_malloc(nTracks * sizeof *js);
  us = safe_malloc(nTracks * sizeof *us);
  fs = safe_malloc(ww * sizeof *fs);

  prune_tracker_states(s, nTracks, roles, top_n, ii, &ed, &ed2);
  /* left-to-right; sparse representation; state tying */
  prune_xHMM_states(s, xhmm->uu, &es, &es2, &es3);

  jj = VLENGTH(ed);
  vv = VLENGTH(es);
  uu = jj * vv;

  final = safe_malloc(xhmm->uu * sizeof *final);
  
  for (u = 0; u < xhmm->uu; u ++){
    final[u] = TRUE;
    if (final_state){
      HMM_decodeXStates(us, hmms, u, ww);
      for (w = 0; w < ww; w ++)
	/* if the state is not at the final, ignore */
	if (us[w] != padded_uu(hmms[w]->uu)-1)
	  break;
      if (w < ww) 
	final[u] = FALSE;
    }
  }
  /* memory allocation */
  outL = safe_malloc(tt * sizeof *outL);
  alpha = safe_malloc(tt * sizeof *alpha);
  alpha_p = safe_malloc(tt * sizeof *alpha_p);
  xh= safe_malloc(tt * sizeof *xh);
  xi = safe_malloc(tt * sizeof *xi);
  beta = safe_malloc(tt * sizeof *beta);
  for (t = 0; t < tt; t ++){
    outL[t] = safe_malloc(uu * sizeof *(outL[t]));
    alpha[t] = safe_malloc(uu * sizeof *(alpha[t]));
    alpha_p[t] = safe_malloc(uu * sizeof *(alpha_p[t]));
    beta[t] = safe_malloc(uu * sizeof *(beta[t]));
    xh[t] = safe_malloc(uu * sizeof *(xh[t]));
    xi[t] = safe_malloc(uu * sizeof *(xi[t]));
    for (u = 0; u < uu; u ++){
      xh[t][u] = safe_malloc(ww * sizeof *(xh[t][u]));
      xi[t][u] = safe_malloc(ww * sizeof *(xi[t][u]));
      for (w = 0; w < ww; w ++){
	xh[t][u][w] = safe_malloc(s->ws[w]->ffm[0]->ii * sizeof *(xh[t][u][w]));
	xi[t][u][w] = safe_malloc(s->ws[w]->uu * sizeof *(xi[t][u][w]));
      }
    }
  }
  /* here we go */
  a = MATRIX(xhmm->A);
  b = VECTOR(xhmm->B);
  sentence_score = NEGATIVE_INFINITY;

  for (j0 = 0; j0 < (DISPLACEMENT?jj:1); j0 ++){
    m0->j0 = j0;
    if (DISPLACEMENT){
      decode_box_states(U(j0), ii, nTracks, js);
      memset(m0->p0, -1, MAX_NTRACKS * sizeof *(m0->p0));
      /* get the initial participant-to-detection mapping */
      for (i = 0; i < nTracks; i ++)
	m0->p0[i] = js[i];
    }
    /* re-compute the likelihood given the new anchor detections p0 */
    compute_crossed_likelihood(ed, es, s, nTracks, roles, phi1, phi2, phi0, tt, ii, outL, xh, m0);
    compute_sentence_alpha(ed, ed2, es, es2, final, s, ii, tt, nTracks, a, b, bps, bss, outL, xa, xb, alpha, alpha_p, xi, m0);
    compute_sentence_beta(ed, ed2, es, es3, final, ii, tt, nTracks, a, bps, bss, outL, beta, m0);    
    /* compute likelihood */
    return_value = NEGATIVE_INFINITY;
    for (u = 0; u < uu; u ++)
      return_value = add_exp(return_value, alpha[tt - 1][u]);
    sentence_score = add_exp(sentence_score, return_value);

    /* compute derivatives with reverse mode; not finished */
    kk = (DISPLACEMENT? VLENGTH(ed2[j0]): jj);
    for (t = 0; t < tt; t ++){
      /* frame t */
      for (j = 0; j < kk; j ++){
	phi = (DISPLACEMENT? J(j,j0): j);
	decode_box_states(U(phi), ii, nTracks, js);
	single = 0;
	for (i = 0; i < nTracks; i ++)
	  single += bss[t][js[i]];
	/* compute features */
	for (w = 0; w < ww; w ++){
	  nRoles = VLENGTH(roles[w]);
	  assert(nRoles <= 3 && nRoles >= 1);
	  r = VECTOR(roles[w]);
	  d1 = js[r[0]];
	  m0->d0[0] = m0->p0[r[0]];
	  d2 = -1;
	  d3 = -1;
	  if (nRoles >= 2){
	    assert(r[0] != r[1]);
	    d2 = js[r[1]];
	    m0->d0[1] = m0->p0[r[1]];
	  }
	  if (nRoles >= 3){
	    assert(r[1] != r[2] && r[0] != r[2]);
	    d3 = js[r[2]];
	    m0->d0[2] = m0->p0[r[2]];
	  }
	  fs[w] = new_feature(s->ws[w]->pos, d1, d2, d3, phi1[t], phi2[t], phi0[t], m0);
	  assert(fs[w]->ii == s->ws[w]->ffm[0]->ii);
	}
	/* compute derivatives */
	for (u = 0; u < vv; u ++){
	  HMM_decodeXStates(us, hmms, W(u), ww);
	  x = u * jj + phi;
	  /* derivatives for initial probability at time 0 */
	  if (t == 0){
	    for (w = 0; w < ww; w ++){
	      nb = VECTOR(SENTENCE_HMM_B(tmp, w));
	      eu = effective_state(us[w]);
	      nb[eu] = add_exp(nb[eu], beta[0][x] + xi[0][x][w][0] + outL[0][x] + single);
	    }
	  }
	  /* derivatives for output distributions */
	  for (w = 0; w < ww; w ++){
	    eu = effective_state(us[w]);
	    for (i = 0; i < fs[w]->ii; i ++){
	      assert(s->ws[w]->ffm[eu]->ft[i] == FT_DISCRETE);
	      np = SENTENCE_FFM_P(tmp, w, eu, i);
	      z = (int)(fs[w]->v[0][i]);
	      assert(z >= 0 && z < np->kk);
	      np->p[z] = add_exp(np->p[z], beta[t][x] + alpha_p[t][x] + xh[t][x][w][i] + single);
	    }
	  }
	  /* derivatives for transition matrix */
	  if (t > 0){
	    for (w = 0; w < ww; w ++){
	      eu = effective_state(us[w]);
	      na = MATRIX(SENTENCE_HMM_A(tmp, w));
	      for (ev = 0; ev < hmms[w]->uu; ev ++)
		na[ev][eu] = add_exp(na[ev][eu], beta[t][x] + xi[t][x][w][ev] + outL[t][x] + single);
	    }
	  }
	}
	for (w = 0; w < ww; w ++)
	  freeFeatures(fs[w]);
      }
    }
  }
  /* No track eligible */
  if (sentence_score == NEGATIVE_INFINITY){
    printf("viterbi-sentence-track-in-c, all costs were infinity\n");
  }
  /* Clean up */
  for (t = 0; t < tt; t ++){
    free(alpha[t]);
    free(alpha_p[t]);
    free(beta[t]);
    free(outL[t]);
    for (u = 0; u < uu; u ++){
      for (w = 0; w < ww; w ++){
	free(xh[t][u][w]);
	free(xi[t][u][w]);
      }
      free(xh[t][u]);
      free(xi[t][u]);
    }
    free(xh[t]);
    free(xi[t]);
  }
  for (j = 0; j < VLENGTH(ed); j ++)
    freeIVec(ed2[j]);
  for (u = 0; u < VLENGTH(es); u ++){
    freeIVec(es2[u]);
    freeIVec(es3[u]);
  }
  for (u = 0; u < xhmm->uu; u ++){
    free(xb[u]);
    for (v = 0; v < xhmm->uu; v ++)
      free(xa[u][v]);
    free(xa[u]);    
  }
  free(hmms);
  free(js);
  free(us);
  free(fs);
  free(final);
  free(outL);
  free(alpha);
  free(alpha_p);
  free(xh);
  free(xa);
  free(xb);
  free(xi);
  free(beta);
  freeIVec(ed);
  freeIVec(es);
  free(ed2);
  free(es2);
  free(es3);
  freeHMM(xhmm);
  free(m0);
  return sentence_score;
}

void update_objective_derivatives(int nn, Real e, Sentence *scratches, Sentence *tmps, Real *likelihoods){
  int n, u, v, uu, w, i, z;
  Real **sa, *sb, **na, *nb, *aver_likes, log_e = my_log(e);
  Real sum_likelihoods = NEGATIVE_INFINITY;
  DParam *np, *sp;
 
  aver_likes = safe_malloc(nn * sizeof *aver_likes);

  for (n = 0; n < nn; n ++){
    aver_likes[n] = likelihoods[n] * e;
    sum_likelihoods = add_exp(sum_likelihoods, aver_likes[n]);
  }
  for (n = 0; n < nn; n ++){
    if (likelihoods[n] == NEGATIVE_INFINITY) // derivatives undefined
      continue;
    for (w = 0; w < tmps[n]->ww; w ++){
      sa = MATRIX(SENTENCE_HMM_A(scratches[n], w));
      sb = VECTOR(SENTENCE_HMM_B(scratches[n], w));    
      na = MATRIX(SENTENCE_HMM_A(tmps[n], w));
      nb = VECTOR(SENTENCE_HMM_B(tmps[n], w));
      uu = tmps[n]->ws[w]->uu;
      for (u = 0; u < uu; u ++){
	for (i = 0; i < OUTPUT_MODEL(tmps[n], w, u)->ii; i ++){
	  np = (DParam *)SENTENCE_FFM_P(tmps[n], w, u, i);
	  sp = (DParam *)SENTENCE_FFM_P(scratches[n], w, u, i);
	  for (z = 0; z < np->kk; z ++){
	    if (nn > 1)
	      sp->p[z] -= my_exp(np->p[z] + (e-1)*likelihoods[n] - sum_likelihoods + log_e);
	    if (n == 0)
	      sp->p[z] += my_exp(np->p[z] - likelihoods[n] + log_e);
	  }
	}
	if (nn > 1)
	  sb[u] -= my_exp(nb[u] + (e-1)*likelihoods[n] - sum_likelihoods + log_e);
	if (n == 0)
	  sb[u] += my_exp(nb[u] - likelihoods[n] + log_e);
	for (v = 0; v < uu; v ++){
	  if (nn > 1)
	    sa[u][v] -= my_exp(na[u][v] + (e-1)*likelihoods[n] - sum_likelihoods + log_e);
	  if (n == 0)
	    sa[u][v] += my_exp(na[u][v] - likelihoods[n] + log_e);
	}
      }
    }
  }
  free(aver_likes);
}

/* -- accessor functions -- */
int model_ii(Model *m)
{ assert(m!=NULL&&m->uu>0);
  return m->ffm[0]->ii;}

int model_nn(Model *m, int i){
  assert(m != NULL);
  if (i < 0 || i >= model_ii(m))
    panic("model_nn: i = %d out of range [0, %d)", i, model_ii(m));
  assert(m->ffm[0]->ft[i] == FT_DISCRETE);
  return ((DParam *)m->ffm[0]->p[i])->kk;
}

int model_feature_type(Model *m, int i)
{ assert(m!=NULL&&m->uu>0);
  if (i<0||i>=model_ii(m))
  { panic("model_feature_type(): i = %d out of range [0, %d)",
	  i, model_ii(m));}
  return m->ffm[0]->ft[i];}

Real model_parameter(Model *m, int u, int i, int n)
{ assert(m!=NULL&&m->uu>0);
  if (i<0||i>=model_ii(m))
  { panic("model_parameter(): i = %d out of range [0, %d]", i, model_ii(m));}
  if (u<0||u>=m->uu)
  { panic("model_parameter(): u = %d out of range [0, %d]", u, m->uu);}
  switch (m->ffm[u]->ft[i])
  { case FT_CONTINUOUS:
    switch (n)
    { case 0: return ((Param *)m->ffm[u]->p[i])->mu;
      case 1: return ((Param *)m->ffm[u]->p[i])->sigma;
      default: panic("model_parameter(): n = %d out of range [0, 1]", n);}
    case FT_RADIAL:
    switch (n)
    { case 0: return ((VMParam *)m->ffm[u]->p[i])->mean;
      case 1: return ((VMParam *)m->ffm[u]->p[i])->kappa;
      default: panic("model_parameter(): n = %d out of range [0, 1]", n);}
    case FT_DISCRETE:
    switch (n)
    { case 0: return ((DParam *)m->ffm[u]->p[i])->kk;
      default:
	if (n < 1 || n > ((DParam *)m->ffm[u]->p[i])->kk)
	  panic("model_parameter(): n = %d out of range [0, %d)", n,
		((DParam *)m->ffm[u]->p[i])->kk);
      return ((DParam *)m->ffm[u]->p[i])->p[n-1];}
    default:
    panic("model_parameter(): Unrecognised feature type: %d",
	  m->ffm[u]->ft[i]);}
  panic("model_parameter(): Control shouldn't reach this point");
  return 0.0;}

int model_uu(Model *m)
{ assert(m!=NULL);
  return m->uu;}

Hmm *model_hmm(Model *m){
  assert(m!=NULL);
  return m->hmm;
}

Ffm *model_ffm(Model *m, int i){
  assert(m!=NULL);
  return m->ffm[i];
}

Real model_a(Model *m, int u, int v)
{ assert(m!=NULL&&m->hmm!=NULL);
  if (u<0||u>=m->uu||v<0||v>=m->uu)
  { panic("model_a(): u = %d or v = %d out of range [0, %d]", u, v, m->uu);}
  return MATRIX(m->hmm->A)[u][v];}

Real model_b(Model *m, int u)
{ assert(m!=NULL&&m->hmm!=NULL);
  if (u<0||u>=m->uu) panic("model_b(): u = %d out of range [0, %d]", u, m->uu);
  return VECTOR(m->hmm->B)[u];}

char *model_name(Model *m)
{ assert(m != NULL);
  return m->name;
}

int model_n_roles(Model *m)
{ assert(m != NULL && m->n_roles > 0);
  return m->n_roles;
}

enum part_of_speech model_pos(Model *m)
{ assert(m != NULL);
  return m->pos;
}

void set_model_parameter(Model *m, int u, int i, int n, Real x)
{ int j;
  assert(m!=NULL&&m->uu>0);
  if (i<0||i>=model_ii(m))
  { panic("set_model_parameter(): i = %d out of range [0, %d]",
	  i, model_ii(m));}
  if (u<0||u>=m->uu)
  { panic("set_model_parameter(): u = %d out of range [0, %d]", u, m->uu);}
  if (m->ffm[u]->p[i]==NULL)
  { panic("set_model_parameter(): feature %d is not initialised", i);}
  switch (m->ffm[u]->ft[i])
  { case FT_CONTINUOUS:
    switch (n)
    { case 0:
      ((Param *)m->ffm[u]->p[i])->mu = x;
      break;
      case 1:
      if (x<=0.0){
	((Param *)m->ffm[u]->p[i])->sigma = 0.01;
      } 
      else { ((Param *)m->ffm[u]->p[i])->sigma = x; }
      break;
      default:
      panic("set_model_parameter(): n = %d out of range [0, 1]", n);}
    break;
    case FT_RADIAL:
    switch (n)
    { case 0:
      ((VMParam *)m->ffm[u]->p[i])->mean = x;
      break;
      case 1:
      if (x<=0.0){
	((VMParam *)m->ffm[u]->p[i])->kappa = 0.01;
      } else { ((VMParam *)m->ffm[u]->p[i])->kappa = x; }
      break;
      default:
      panic("set_model_parameter(): n = %d out of range [0, 1]", n);}
    break;
    case FT_DISCRETE:
    switch (n)
    { case 0:
      j = (int)round(x);
      if (j<=0||j>MAX_DISCRETE)
	panic("set_model_parameter(): x = %f is out of range [1, MAX_DISCRETE]", x);
      ((DParam *)m->ffm[u]->p[i])->kk = j;
      break;
      default:
      if (n-1<0||n-1>=((DParam *)m->ffm[u]->p[i])->kk)
	panic("model_parameter(): n-1 = %d out of range [0, kk - 1]", n-1);
      ((DParam *)m->ffm[u]->p[i])->p[n-1] = x; }
    break;
    default:
    panic("set_model_paramter(): Unrecognised feature type: %d, %d, %d",
	  m->ffm[u]->ft[i], u, i);}}

void set_model_a(Model *m, int u, int v, Real x)
{ assert(m!=NULL&&m->hmm!=NULL);
  if (u<0||u>=m->uu||v<0||v>=m->uu)
  { panic("set_model_a(): u = %d or v = %d out of range [0, %d]",
	  u, v, m->uu);}
  MATRIX(m->hmm->A)[u][v] = x;}

void set_model_b(Model *m, int u, Real x)
{ assert(m!=NULL&&m->hmm!=NULL);
  if (u<0||u>=m->uu)
  { panic("set_model_b(): u = %d out of range [0, %d]", u, m->uu);}
  VECTOR(m->hmm->B)[u] = x;}

void set_model_name(Model *m, char *name)
{ assert(m != NULL && name != NULL);
  if (m->name) free(m->name);
  m->name = safe_malloc(strlen(name) + 1);
  strcpy(m->name, name);
}

void set_model_n_roles(Model *m, int n)
{ assert(m != NULL);
  if (n < 0)
    panic("set_model_n_roles: n = %d is negative", n);
  m->n_roles = n;
}

void set_model_pos(Model *m, enum part_of_speech pos)
{ assert(m != NULL);
  m->pos = pos;
}

char *model_pos_str(enum part_of_speech pos)
{
  switch(pos){
  case NOUN: return "noun";
  case VERB: return "verb";
  case ADVERB: return "adverb";
  case ADJECTIVE: return "adjective";
  case PREPOSITION: return "preposition";
  case MOTION_PREPOSITION: return "motion_preposition";
  default: ;
  }
  return "other";
}

Real single_hmm_run(int **data, int tt, Model *m, Model *scratch, int baum_welch){
  int uu, t, u, v, z;
  uu = m->uu;
  Real gamma[tt][uu], psi[tt][uu][uu], alpha[tt][uu], alpha_p[tt][uu];
  Real beta[tt][uu], outL[tt][uu], map, norm;
  RVec feature = allocRVec(1);
  Real **a, *b, **na, *nb, **sa, *sb;  
  DParam *np, *sp;
  Model *tmp = cloneModel(scratch);
  zeroModel(tmp);
  linear2logModel(tmp);

  a = MATRIX(m->hmm->A);
  b = VECTOR(m->hmm->B);
  /* output likelihood */
  for (t = 0; t < tt; t ++)
    for (u = 0; u < uu; u ++){
      VECTOR(feature)[0] = data[t][0];
      outL[t][u] = FFM_likelihood_one(m->ffm[u], feature, NULL);
    }
  /* alpha */
  for (u = 0; u < uu; u ++){
    alpha_p[0][u] = my_log(b[u]);
    alpha[0][u] = alpha_p[0][u] + outL[0][u];
    
  }
  for (t = 1; t < tt; t ++)
    for (u = 0; u < uu; u ++){
      alpha_p[t][u] = NEGATIVE_INFINITY;
      for (v = 0; v < uu; v ++)
	alpha_p[t][u] = add_exp(alpha_p[t][u], alpha[t-1][v] + my_log(a[v][u]));
      alpha[t][u] = alpha_p[t][u] + outL[t][u];
    }
  map = NEGATIVE_INFINITY;
  for (u = 0; u < uu; u ++)
    map = add_exp(map, alpha[tt-1][u]);
  /* beta */
  for (u = 0; u < uu; u ++)
    beta[tt-1][u] = 0;
  for (t = tt-2; t >= 0; t --)
    for (u = 0; u < uu; u ++){
      beta[t][u] = NEGATIVE_INFINITY;
      for (v = 0; v < uu; v ++)
	beta[t][u] = add_exp(beta[t][u], beta[t+1][v] + my_log(a[u][v]) + outL[t+1][v]);      
    }

  if (baum_welch){
    /* gamma */
    for (t = 0; t < tt; t ++){
      norm = NEGATIVE_INFINITY;
      for (u = 0; u < uu; u ++){
	gamma[t][u] = alpha[t][u] + beta[t][u];
	norm = add_exp(norm, gamma[t][u]);
      }
      /* normalize */
      for (u = 0; u < uu; u ++)
	gamma[t][u] -= norm;
    }
    /* psi */
    for (t = 0; t < tt-1; t ++){
      norm = NEGATIVE_INFINITY;
      for (u = 0; u < uu; u ++)
	for (v = 0; v < uu; v ++){
	  psi[t][u][v] = alpha[t][u] + my_log(a[u][v]) + beta[t+1][v] + outL[t+1][v];
	  norm = add_exp(norm, psi[t][u][v]);
	}
      for (u = 0; u < uu; u ++)
	for (v = 0; v < uu; v ++)
	  psi[t][u][v] -= norm;
    }
    /* update output distribution */
    for (t = 0; t < tt; t ++)
      for (u = 0; u < uu; u ++){
	z = (int)(data[t][0]);
	((DParam *)(scratch->ffm[u]->p[0]))->p[z] += my_exp(gamma[t][u]);
      }
    /* update transition */
    for (t = 0; t < tt - 1; t ++)
      for (u = 0; u < uu; u ++){
	if (t == 0)
	  VECTOR(scratch->hmm->B)[u] += my_exp(gamma[0][u]);
	for (v = 0; v < uu; v ++)
	  MATRIX(scratch->hmm->A)[u][v] += my_exp(psi[t][u][v]);
      }
  }
  /* derivatives */
  else{
    /* log space */
    na = MATRIX(tmp->hmm->A);
    nb = VECTOR(tmp->hmm->B);
    for (t = tt-1; t >= 0; t --){
      z = (int)(data[t][0]);
      for (v = 0; v < uu; v ++){
	if (t == 0)
	  nb[v] = add_exp(nb[v], beta[0][v] + outL[0][v]);	
	np = (DParam *)(tmp->ffm[v]->p[0]); 
	np->p[z] = add_exp(np->p[z], beta[t][v] + alpha_p[t][v]);
	for (u = 0; u < uu; u ++)
	  if (t > 0)
	    na[v][u] = add_exp(na[v][u], beta[t][u] + alpha[t-1][v] + outL[t][u]);
      }
    }
    /* df -> dlogf */
    sa = MATRIX(scratch->hmm->A);
    sb = VECTOR(scratch->hmm->B);    
    for (u = 0; u < uu; u ++){
      np = (DParam *)(tmp->ffm[u]->p[0]);
      sp = (DParam *)(scratch->ffm[u]->p[0]);
      for (z = 0; z < np->kk; z ++)
	sp->p[z] += my_exp(np->p[z] - map);
      sb[u] += my_exp(nb[u] - map);
      for (v = 0; v < uu; v ++)
	sa[u][v] += my_exp(na[u][v] - map);
    }    
  }
  freeModel(tmp);
  freeRVec(feature);
  return map;
}

void gt_with_derivatives_ml(Model *m, Model *scratch){
  Real **a, *b, **na, *nb;  
  DParam *np, *p;
  int u, uu = m->uu, z, v;
  a = MATRIX(m->hmm->A);
  b = VECTOR(m->hmm->B);
  na = MATRIX(scratch->hmm->A);
  nb = VECTOR(scratch->hmm->B);
  for (u = 0; u < uu; u ++){
      np = (DParam *)(scratch->ffm[u]->p[0]);
      p = (DParam *)(m->ffm[u]->p[0]);
      for (z = 0; z < p->kk; z ++)
	np->p[z] *= p->p[z];
      nb[u] *= b[u];
      for (v = 0; v < uu; v ++)
	na[u][v] *= a[u][v];
    }
}

/* Tam V'Nishlam Shevah L'El Borei Olam */
