/* LaHaShem HaAretz U'Mloah */

/* Program to train and evaluate hidden Markov models */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "hmm.h"
#include "hmm-rand.h"
#include "hmm-data.h"
#include "hmm-control.h"

static RMat g_alpha = NULL;
static RMat g_delta = NULL;
static RMat g_deltav = NULL;
static RMat g_beta = NULL;
static RMat3d g_psi = NULL;

extern int MIN_DURATION;
extern int DURATION;
extern int PAD_ALL;

static void panic (const char *error_text, ...)
{ 
  va_list ap;
  char error_message[1024];
  va_start(ap, error_text);
  vsprintf(error_message, error_text, ap);
  fprintf(stderr, "panic: %s\n", error_message);
  exit(EXIT_FAILURE);
}

int padded_uu(int uu){
  /* don't pad single-state hmms */
  if (DURATION && uu > 1){
    if (!PAD_ALL)
      uu = MIN_DURATION * (uu - 2) + 2; /* only intermediate states are padded */
    else
      uu *= MIN_DURATION;
  }
  return uu;
}

int effective_state(int u){
  if (DURATION && u > 0){
    if (!PAD_ALL)
      u = (u - 1) / MIN_DURATION + 1;
    else
      u /= MIN_DURATION;
  }
  return u;  
}

/* allocHMM - Allocates space for an HMM and returns a pointer to it. */
Hmm *allocHMM(int uu){ 
  Hmm *m = safe_malloc(sizeof *m);
  assert(uu > 0);
  m->A = allocRMat(uu, uu);
  m->B = allocRVec(uu);
  m->uu = uu;
  return m;
}

/* constantHMM -
 *  Initialises an HMM with constant transition log probabilities.
 */
void constantHMM(Hmm *m, int upper_triangular){ 
  int u, v;
  zeroHMM(m);
  for (u = 0; u < m->uu; u++){ 
    VECTOR(m->B)[u] = 1;
    for (v = upper_triangular? u: 0; v < m->uu; v++) 
      MATRIX(m->A)[u][v] = 1;
  }
  normaliseHMM(m, upper_triangular);
}

void copyHMM(Hmm *m1, Hmm *m2){ 
  int u;
  if(m1->uu != m2->uu){
    freeHMM(m1);
    m1 = allocHMM(m2->uu);
  }
  copyRVec(VECTOR(m1->B), VECTOR(m2->B), m1->uu);
  for (u = 0; u<m1->uu; u++)
    copyRVec(MATRIX(m1->A)[u], MATRIX(m2->A)[u], m1->uu);
}

/* displayHMM - Writes the HMM `m' in a human-readable format to output
 *  stream `f'.  Only displays a maximum of DISPLAY_HMM_MAX_U states
 *  on a single line.
 */
void displayHMM(void *p, Hmm *m){ 
  int i, u;
  FILE *f = (FILE *)p;
  /* display B vector */
  fprintf(f, "%8s", "");
  for (u = 0; u < DISPLAY_HMM_MAX_U && u < m->uu; u++) 
    fprintf(f, " %4d", u);
  fprintf(f, "\n");
  fprintf(f, "HMM:  B:");
  for (u = 0; u < DISPLAY_HMM_MAX_U && u < m->uu; u++)
    fprintf(f, " %4.2f", VECTOR(m->B)[u]);
  fprintf(f, "\n");
  for (;u<m->uu; u++){ 
    fprintf(f, "\n");
    fprintf(f, "%8s", "");
    for (i = 0; i < DISPLAY_HMM_MAX_U && i + u < m->uu; u++, i++)
      fprintf(f, " %4d", i + u);
    fprintf(f, "\n");
    for (i = 0; i < DISPLAY_HMM_MAX_U && u < m->uu; u++, i++)
      fprintf(f, " %4.2f", VECTOR(m->B)[u]);
    fprintf(f, "\n");}
  /* display A matrix */
  fprintf(f, "%8s", "A:");
  for (u = 0; u < DISPLAY_HMM_MAX_U && u < m->uu; u++) 
    fprintf(f, " %4d", u);
  fprintf(f, "\n");
  for (u = 0; u < DISPLAY_HMM_MAX_U && u < m->uu; u++){ 
    fprintf(f, "%8d", u);
    for (i = 0; i < DISPLAY_HMM_MAX_U && i < m->uu; i++) 
      fprintf(f, " %4.2f", MATRIX(m->A)[u][i]);
    fprintf(f, "\n");
  }
}

/* freeHMM - Frees a previously allocated HMM. */
void freeHMM(Hmm *m){ 
  freeRMat(m->A);
  freeRVec(m->B);
  free(m);
}

/* normaliseHMM - return models in linear space
 */
int normaliseHMM(Hmm *m, int upper_triangular){ 
  int u, v, w;
  Real sum;
  sum = 0.0;
  for (u = 0; u < m->uu; u++)
    sum += VECTOR(m->B)[u];
  if (!(fabs(sum) > 0.0))
    printf("sum error: %e\n", sum);
  assert(fabs(sum) > 0.0);
  for (u = 0; u < m->uu; u++) {
    if(VECTOR(m->B)[u] / sum < -1e-50){
      printf("normaliseHMM: b error\n");
      return FALSE;
    }
    VECTOR(m->B)[u] /= sum;
  }
  for (u = 0; u < m->uu; u++){ 
    sum = 0.0;
    w = (upper_triangular?u:0);
    for (v = w; v < m->uu; v++) 
      sum += MATRIX(m->A)[u][v];
    if (sum==0.0) {
      /* A state never jumps to another state (include itself)
	 DO NOTHING. We don't want a corrected uniform distribution. */
      for(v = 0; v < m->uu; v ++)
	MATRIX(m->A)[u][v] = 0;
      continue;
    }
    for (v = 0; v<m->uu; v ++) {
      if(MATRIX(m->A)[u][v]/sum < -1e-50){
	printf("normaliseHMM: a error\n");
	return FALSE;
      }
      MATRIX(m->A)[u][v] /= sum;
    }
  }
  return TRUE;
}

/* randomiseHMM -
 *  Initialises an HMM with normalised random transition log probabilities.
 */
void randomiseHMM(Hmm *m, int upper_triangular){ 
  int u, v;
  zeroHMM(m);
  if (DURATION)
    assert(upper_triangular);
  for (u = 0; u < m->uu; u++){ 
    if (DURATION && u != 0)
      VECTOR(m->B)[u] = 0;
    else
      VECTOR(m->B)[u] = RANDOM;
    for (v = (upper_triangular?u:0); v < (DURATION?MIN(u+2,m->uu):m->uu); v++){
      MATRIX(m->A)[u][v] = RANDOM;
    }
  }
  normaliseHMM(m, upper_triangular);
}

/* Add noise to m */
void noiseHMM(Hmm *m, int upper_triangular, Real delta){
  int u, v;
  for (u = 0; u < m->uu; u ++){
    VECTOR(m->B)[u] += delta * RANDOM;
    for (v = (upper_triangular? u: 0); v < m->uu; v ++){
      MATRIX(m->A)[u][v] += delta * RANDOM;
    }
  }
  normaliseHMM(m, upper_triangular);
}

void smoothHMM(Hmm *m, int upper_triangular, Real eps){
  int u, v;
  for (u = 0; u < m->uu; u ++){
    VECTOR(m->B)[u] += eps;
    for (v = (upper_triangular? u: 0); v < m->uu; v ++)
      MATRIX(m->A)[u][v] += eps;
  }
  normaliseHMM(m, upper_triangular);
}

/* zeroHMM - Sets all of the transition probabilities to 0.0. */
void zeroHMM(Hmm *m){ 
  int u, v;
  for (u = 0; u<m->uu; u++){ 
    VECTOR(m->B)[u] = 0.0;
    for (v = 0; v<m->uu; v++)
      MATRIX(m->A)[u][v] = 0.0;
  }
}

void linear2logHMM(Hmm *m){
  int u, v;
  for (u = 0; u < m->uu; u ++){
    VECTOR(m->B)[u] = my_log(VECTOR(m->B)[u]);
    for (v = 0; v < m->uu; v ++)
      MATRIX(m->A)[u][v] = my_log(MATRIX(m->A)[u][v]);
  }
}

void log2linearHMM(Hmm *m){
  int u, v;
  for (u = 0; u < m->uu; u ++){
    VECTOR(m->B)[u] = my_exp(VECTOR(m->B)[u]);
    for (v = 0; v < m->uu; v ++)
      MATRIX(m->A)[u][v] = my_exp(MATRIX(m->A)[u][v]);
  }
}

int isZeroHMM(Hmm *m){
  int u, v, zero = 1;
  for (u = 0; u < m->uu; u ++){
    if (VECTOR(m->B)[u] != 0){
      zero = 0;
      break;
    }
    for (v = 0; v < m->uu; v ++){
      if (MATRIX(m->A)[u][v] != 0){
	zero = 0;
	break;
      }
    }
    if (zero == 0) break;
  }
  return zero;
}

/* HMM_xStateRange -
   Calculate the total number of states after crossing all the hmms' states
 */
int HMM_xStateRange(Hmm **hmms, int ww)
{ int w, tot = 1;
  for(w = 0; w < ww; w ++)
    tot *= padded_uu(hmms[w]->uu);
  return tot;
}

/* HMM_decodeXStates -
   Given an xState, decode it into several states, each state for each hmm
 */

void HMM_decodeXStates(int *us, Hmm **hmms, int u, int ww)
{ int w, uu;
  for(w = ww - 1; w >= 0; w --){
    uu = padded_uu(hmms[w]->uu);
    us[w] = u % uu;
    u /= uu;
  }
}

/* HMM_xB - log space
   Calculate the crossed 'b' value (initial probability)
 */
Real HMM_xB(Hmm **hmms, int u, int ww, Real *x)
{
  int us[ww];
  int w, _inf_idx = 0, all_first_states = 1;
  Real tot = 0, b[ww];
  assert(ww < INTEGER_BITS);
  HMM_decodeXStates(us, hmms, u, ww);
  for (w = 0; w < ww; w ++){
    /* pi = (1 0 0 ... 0) hard-coded for duration model */
    if (!DURATION || us[w] == 0)
      b[w] = my_log(VECTOR(hmms[w]->B)[effective_state(us[w])]);
    else{
      b[w] = NEGATIVE_INFINITY;
      all_first_states = 0;
    }
    if (b[w] == NEGATIVE_INFINITY)
      _inf_idx |= (1 << w);
    else  
      tot += b[w];
    if (x) x[w] = NEGATIVE_INFINITY;
  }
  if (x){
    for (w = 0; w < ww; w ++)
      if ((_inf_idx & ~(1 << w)) == 0){ /* if no other b[w'] = -inf */
	x[w] = tot;
	if (!_inf_idx) x[w] -= b[w];
      }
  }
  if (_inf_idx) /* if any b[w] = -inf */
    tot = NEGATIVE_INFINITY;  
  if (DURATION && all_first_states) 
    tot = 0; /* normalize to 1 */
  return tot;
}

/* HMM_xA - log space
   Calculate the crossed 'a' value (transition probability)
   needs work
*/

Real HMM_xA(Hmm **hmms, int u, int v, int ww, Real *x)
{
  int us[ww], vs[ww];
  int w, uu, boundary, eu, ev, _inf_idx = 0;
  Real tot = 0, a[ww];
  assert(ww < INTEGER_BITS);
  HMM_decodeXStates(us, hmms, u, ww);
  HMM_decodeXStates(vs, hmms, v, ww);
  if (PAD_ALL)
    boundary = MIN_DURATION - 1;
  else 
    boundary = 0;  
  for (w = 0; w < ww; w ++){
    uu = padded_uu(hmms[w]->uu);
    eu = effective_state(us[w]);
    ev = effective_state(vs[w]);
    /* inside the padding: i transit to i+1 with p=1 */
    if (DURATION && us[w] % MIN_DURATION != boundary && vs[w] == us[w] + 1)
      a[w] = 0;  // dummy probability
    /* from regular state to itself or next padded state 
       i.e., left-to-right HMM
     */
    else if (!DURATION ||
	     ((us[w] % MIN_DURATION == boundary || us[w] == uu - 1)
	      && (vs[w] == us[w] || vs[w] == us[w] + 1)))
      a[w] = my_log(MATRIX(hmms[w]->A)[eu][ev]);
    else 
      a[w] = NEGATIVE_INFINITY;    
    if (a[w] == NEGATIVE_INFINITY)
      _inf_idx |= (1 << w);
    else 
      tot += a[w];
    if (x) x[w] = NEGATIVE_INFINITY;
  }      
  if (x){
    for (w = 0; w < ww; w ++)
      if ((_inf_idx & ~(1 << w)) == 0){ /* if no other a[w'] = -inf */
	x[w] = tot;
	if (!_inf_idx) x[w] -= a[w];
      }
  }
  if (_inf_idx) /* if any a[w] = -inf */
    tot = NEGATIVE_INFINITY;
  return tot;
}

/* Allocate and initialize a crossed HMM for a sentence with ww hmms
   Also precompute the box pair and single costs for the boxes movie
*/
Hmm *initializeXHMM(Hmm **hmms, int ww, Real ****xa, Real ***xb){
  int uu, u, v;
  Real **a, *b, ***ta = NULL, **tb = NULL;
  Hmm *xhmm;
  uu = HMM_xStateRange(hmms, ww);
  if (xa){
    ta = safe_malloc(uu * sizeof *ta);
    tb = safe_malloc(uu * sizeof *tb);
    for (u = 0; u < uu; u ++){
      ta[u] = safe_malloc(uu * sizeof *(ta[u]));
      tb[u] = safe_malloc(ww * sizeof *(tb[u]));
      for (v = 0; v < uu; v ++)
	ta[u][v] = safe_malloc(ww * sizeof *(ta[u][v]));	
    }
  }
  xhmm = allocHMM(uu);
  /* Compute the crossed HMM initial and transition probability */
  a = MATRIX(xhmm->A);
  b = VECTOR(xhmm->B);
  for (u = 0; u < uu; u ++){
    b[u] = HMM_xB(hmms, u, ww, tb?tb[u]:NULL);
    for (v = 0; v < uu; v ++)
      a[u][v] = HMM_xA(hmms, u, v, ww, ta?ta[u][v]:NULL);
  }
  if (xa){
    *xa = ta;
    *xb = tb;
  }
  return xhmm;
}

/* -- processing functions -- */

/* HMM_calcAlphas -
 *  Given an HMM and a vector of log likelihoods for states assigned to
 *  data vectors in the sequence, calculates the alpha values from the
 *  forward-backward algorithm.
 */
void HMM_calcAlphas(Hmm *m, RVec *logL){ 
  int t, u, v, tt = VLENGTH(logL[0]);
  Real **al, **a = MATRIX(m->A), *b = VECTOR(m->B);
  HMM_initGlobals(m->uu, tt);
  al = MATRIX(g_alpha);
  /* alpha[u][1] = b[u]*logL[u][1]
   * alpha[u][t] = sum(v = 1 ... m->uu, a[v][u]*alpha[v][t-1])*logL[u][t]
   */
  for (u = 0; u < m->uu; u++)
    al[u][0] = VECTOR(logL[u])[0] + my_log(b[u]);
  for (t = 1; t < tt; t++){ 
    for (u = 0; u < m->uu; u++){ 
      al[u][t] = NEGATIVE_INFINITY;
      for (v = 0; v < m->uu; v++)
	al[u][t] = add_exp(al[u][t], my_log(a[v][u]) + al[v][t-1]);
      al[u][t] += VECTOR(logL[u])[t];
    }
  }
}

/* HMM_calcDeltas -
 *  Given an HMM and a vector of log likelihoods for states assigned to
 *  data vectors in the sequence, calculates the delta values from the
 *  forward-backward algorithm.
 */
void HMM_calcDeltas(Hmm *m, RVec *logL){ 
  int t, u, v, tt = VLENGTH(logL[0]);
  /* dev could be int. */
  Real **de, **dev, **a = MATRIX(m->A), *b = VECTOR(m->B), d;
  HMM_initGlobals(m->uu, tt);
  de = MATRIX(g_delta);
  dev = MATRIX(g_deltav);	
  /* delta[u][1] = b[u]*logL[u][1]
   * delta[u][t] = max(v = 1 ... m->uu, a[v][u]*delta[v][t-1])*logL[u][t]
   */
  for (u = 0; u < m->uu; u++) 
    de[u][0] = VECTOR(logL[u])[0] + my_log(b[u]);
  for (t = 1; t < tt; t++){ 
    for (u = 0; u < m->uu; u++){ 
      de[u][t] = NEGATIVE_INFINITY;
      dev[u][t] = 0;		
      for (v = 0; v < m->uu; v++){ 
	d = my_log(a[v][u]) + de[v][t-1];
	if (d > de[u][t]){ 
	  de[u][t] = d;
	  dev[u][t] = v;
	}
      }	
      de[u][t] += VECTOR(logL[u])[t];
    }
  }
}

/* HMM_initGlobals -
 *  Given a number of states and a sequence length, allocates or reallocates
 *  memory for internal variables.
 */
void HMM_initGlobals(int uu, int tt){ 
  if (g_alpha == NULL || COLUMNS(g_alpha) < tt || ROWS(g_alpha) < uu){ 
    if (g_alpha != NULL){ 
      freeRMat(g_alpha);
      freeRMat(g_delta);
      freeRMat(g_deltav);	
      freeRMat(g_beta);
      freeRMat3d(g_psi);
    }
    g_alpha = allocRMat(uu, tt);
    g_delta = allocRMat(uu, tt);
    g_deltav = allocRMat(uu, tt); 
    g_beta = allocRMat(uu, tt);
    g_psi = allocRMat3d(uu, uu, tt);
  }
}

/* HMM_logL -
 *  Given an HMM and a vector of log likelihoods for states assigned to
 *  data vectors in the sequence, returns the log likelihood of the data
 *  given the HMM.
 */
Real HMM_logL(Hmm *m, RVec *logL){ 
  int u, tt = VLENGTH(logL[0]);
  Real l = NEGATIVE_INFINITY;
  /* P(sequence|m) = sum(u = 1 ... m->uu, alpha[u][tt]) */
  HMM_calcAlphas(m, logL);
  for (u = 0; u<m->uu; u++) 
    l = add_exp(l, MATRIX(g_alpha)[u][tt-1]);
  return l;
}

/* HMM_best_state_sequence -
 *  Given an HMM and a vector of log likelihoods for states assigned to
 *  data vectors in the sequence, returns the best state sequence of the data
 *  given the HMM.
 */
int *HMM_best_state_sequence(Hmm *m, RVec *logL){ 
  int u, t, tt = VLENGTH(logL[0]), *rv;
  Real b, d;
  rv = safe_malloc(tt * sizeof *rv);
  HMM_calcDeltas(m, logL);
  b = NEGATIVE_INFINITY;
  rv[tt-1] = 0;
  for (u = 0; u<m->uu; u++){ 
    d = MATRIX(g_delta)[u][tt-1];
    if (d>b){ 
      b = d;
      rv[tt-1] = u;
    }
  }
  for (t = tt-2; t>=0; t--) rv[t] = MATRIX(g_deltav)[rv[t+1]][t+1];
  return rv;
}

/* HMM_calcGammas computes gammas, which are the probabilities that
 * each frame is in each state
 */
void HMM_calcGammas(Hmm *m, RVec *logL, RVec *gamma) {
  int t, u, v, tt = VLENGTH(logL[0]);
  Real **a = MATRIX(m->A), **al, **be;
  Real like;
  assert(VLENGTH(logL[0])==VLENGTH(gamma[0]));
  HMM_initGlobals(m->uu, tt);
  al = MATRIX(g_alpha);
  be = MATRIX(g_beta);
  /* calculate alpha's */
  HMM_calcAlphas(m, logL);
  /* calculate beta's -
   * beta[u][tt] = 1
   * beta[u][t] = sum(v = 1 ... m->uu, a[u][v]*beta[v][t+1]*logL[v][t+1])
   */
  for (u = 0; u<m->uu; u++) be[u][tt-1] = 0.0;
  for (t = tt-2; t>=0; t--){ 
    for (u = 0; u<m->uu; u++){ 
      be[u][t] = NEGATIVE_INFINITY;
      for (v = 0; v<m->uu; v++)
	be[u][t] = add_exp(be[u][t], my_log(a[u][v])+be[v][t+1]+VECTOR(logL[v])[t+1]);
    }
  }
  /* calculate logL of sequence -
   * P(sequence|m) = sum(u = 1 ... m->uu, alpha[u][tt])
   */
  like = NEGATIVE_INFINITY;
  for (u = 0; u<m->uu; u++) like = add_exp(like, al[u][tt-1]);
  /* calculate responsibilities
   *               alpha[u][t]*beta[u][t]
   * gamma[u][t] = ----------------------
   *                    P(data|model)
   */
  for (t = 0; t<tt; t++)
    for (u = 0; u<m->uu; u++) 
      VECTOR(gamma[u])[t] = al[u][t]+be[u][t]-like;
}

/* HMM_updateModel -
 *  Given an HMM and a vector of log likelihoods for states in the sequences,
 *  calculates the responsibilities of each state in the HMM for each symbol
 *  in the sequences, and maximises the model parameters based on the
 *  assigned log likelihoods.
*/
Real HMM_updateModel(Hmm *m, Hmm *new_m, RVec *logL, RVec *gamma, Real log_D,
		     Real postpC, int c, int c_ls,
		     enum training_mode training_mode) {
  int t, u, v, tt = VLENGTH(logL[0]);
  Real **a = MATRIX(m->A), *b = VECTOR(m->B), **al, **be, ***ps;
  Real logD = 0, like, dtf;
  int Sc = (c==c_ls);
  switch (training_mode){
  case HMM_ML:
    assert(postpC==0.0);
    logD = NEGATIVE_INFINITY;
    break;
  case HMM_DT:
    assert(c>=0&&c_ls>=0);
    logD = log_D;
    break;
  default: panic("unrecognized training mode");
  }
  assert(VLENGTH(logL[0])==VLENGTH(gamma[0]));
  HMM_initGlobals(m->uu, tt);
  al = MATRIX(g_alpha);
  be = MATRIX(g_beta);
  ps = MATRIX(g_psi);
  /* calculate alpha's */
  HMM_calcAlphas(m, logL);
  /* calculate beta's -
   * beta[u][tt] = 1
   * beta[u][t] = sum(v = 1 ... m->uu, a[u][v]*beta[v][t+1]*logL[v][t+1])
   */
  for (u = 0; u<m->uu; u++) be[u][tt-1] = 0.0;
  for (t = tt-2; t>=0; t--){ 
    for (u = 0; u<m->uu; u++){ 
      be[u][t] = NEGATIVE_INFINITY;
      for (v = 0; v<m->uu; v++)
	be[u][t] = add_exp(be[u][t], my_log(a[u][v])+be[v][t+1]+VECTOR(logL[v])[t+1]);
    }
  }
  /* calculate logL of sequence -
   * P(sequence|m) = sum(u = 1 ... m->uu, alpha[u][tt])
   */
  like = NEGATIVE_INFINITY;
  for (u = 0; u<m->uu; u++)
    like = add_exp(like, al[u][tt-1]);

  /* A sample that can NEVER belong to this category */
  if(like == NEGATIVE_INFINITY){
    assert(postpC == 0.0);
    assert(Sc==0);
  }

  /* calculate responsibilities
   *               alpha[u][t]*beta[u][t]
   * gamma[u][t] = ----------------------
   *                    P(data|model)
   */
  for (t = 0; t<tt; t++){
     for (u = 0; u<m->uu; u++){
       if(like!=NEGATIVE_INFINITY)
	 VECTOR(gamma[u])[t] = al[u][t]+be[u][t]-like;
       else
	 VECTOR(gamma[u])[t] = NEGATIVE_INFINITY;
     }
  }
  /* calculate time-indexed transition probabilities
   *                alpha[u][t]*a[u][v]*logL[v][t+1]*beta[v][t+1]
   * psi[u][v][t] = ---------------------------------------------
   *                               P(data|model)
   */
  for (u = 0; u<m->uu; u++){
    for (v = 0; v<m->uu; v++){
      for (t = 0; t<tt-1; t++){
	if(like!=NEGATIVE_INFINITY)
	  ps[u][v][t] = al[u][t]+my_log(a[u][v])+VECTOR(logL[v])[t+1]+be[v][t+1]-like;
	else
	  ps[u][v][t] = NEGATIVE_INFINITY;
      }
    }
  }
  /* Update new model. The model may have been partly updated by some training
     samples. */
  a = MATRIX(new_m->A);
  b = VECTOR(new_m->B);
  /* calculate B
     b[u] = gamma[u][1]
     - added scaling by sum of gammas to catch any numerical accuracy problems
     not log space here
   */
  for (u = 0; u<m->uu; u++){
    /* This may be negative */
    b[u] += (Sc-postpC)*my_exp(VECTOR(gamma[u])[0])
      +my_exp(logD)*VECTOR(m->B)[u];
  }
  /* calculate A matrix
   *                    sum(t = 1 ... tt-1, psi[u][v][t])
   * a[u][v] = -------------------------------------------------------
   *           sum(t = 1 ... tt-1, sum(w = 1 ... m->uu, psi[u][w][t]))
   * see note above about log space
   */
  for (u = 0; u<m->uu; u++){
    for (v = 0; v<m->uu; v++){
      /* This may be negative */
      dtf = 0.0;
      for(t = 0; t<tt-1; t++)
	dtf += my_exp(ps[u][v][t])*(Sc-postpC) + my_exp(logD)*MATRIX(m->A)[u][v];
      a[u][v] += dtf;
    }
  }
  for (t = 0; t<tt; t++){
    for (u = 0; u<m->uu; u++) VECTOR(gamma[u])[t] = my_exp(VECTOR(gamma[u])[t]);
  }
  return like;
}

/* Tam V'Nishlam Shevah L'El Borei Olam */
