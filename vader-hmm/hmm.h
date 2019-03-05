/* LaHaShem HaAretz U'Mloah */

#ifndef HMM_H
#define HMM_H

#include "hmm-def.h"
#include "hmm-features.h"

#define HMM_STATES(x) ((x)->uu)

#define DISPLAY_HMM_MAX_U 10

typedef struct {
  /* Haonan: the reprensation has been changed back to linear space 
     This does not really cause trouble if you do want to calculate HMM statistics in log space,
     (just covert them to log space when calculating using function 'linear2logHMM')
     The underflow problem potentially happens only during the calculation of alpha, beta, psi,
     not when estimating or manipulating (normalizing, etc) the parameters 
     The benefits of linear space representation:
     1. coherent with the representation space of output distributions
     2. there is another way of avoiding underflow problem even in linear space
   */
  RMat A;
  RVec B;
  int uu;
} Hmm;

Hmm *allocHMM(int uu);

void constantHMM(Hmm *m, int upper_triangular);

void copyHMM(Hmm *m1, Hmm *m2);

void displayHMM(void *p, Hmm *m);

void freeHMM(Hmm *m);

int normaliseHMM(Hmm *m, int upper_triangular);

void randomiseHMM(Hmm *m, int upper_triangular);

void noiseHMM(Hmm *m, int upper_triangular, Real delta);

void smoothHMM(Hmm *m, int upper_triangular, Real eps);

void zeroHMM(Hmm *m);

void linear2logHMM(Hmm *m);

void log2linearHMM(Hmm *m);

int isZeroHMM(Hmm *m);

int HMM_xStateRange(Hmm **hmms, int ww);

void HMM_decodeXStates(int *us, Hmm **hmms, int u, int ww);

Real HMM_xB(Hmm **hmms, int u, int ww, Real *x);

Real HMM_xA(Hmm **hmms, int u, int v, int ww, Real *x);

Hmm *initializeXHMM(Hmm **hmms, int ww, Real ****xa, Real ***xb);

void HMM_calcAlphas(Hmm *m, RVec *logL);

void HMM_calcDeltas(Hmm *m, RVec *logL);

void HMM_calcGammas(Hmm *m, RVec *logL, RVec *gamma);

void HMM_initGlobals(int uu, int tt);

Real HMM_logL(Hmm *m, RVec *logL);

int *HMM_best_state_sequence(Hmm *m, RVec *logL);

Real HMM_updateModel
(Hmm *m, Hmm *new_m, RVec *logL, RVec *gamma, Real log_D, 
 Real postpC, int c, int c_ls, enum training_mode training_mode);

int padded_uu(int uu);

int effective_state(int u);

#endif

/* Tam V'Nishlam Shevah L'El Borei Olam */
