/* LaHaShem HaAretz U'Mloah */

#ifndef HMM_CONTROL_H
#define HMM_CONTROL_H

#include "hmm-def.h"
#include "hmm-data.h"

typedef struct {
  Hmm *hmm;			
  Ffm **ffm;			
  int uu;			/* number of states in HMM */
  char *name;                   /* HMM name */             
  int n_roles;                  /* Number of roles associated with the HMM for a word in a sentence */
  enum part_of_speech pos;      /* Part-of-speech of the word the HMM associated with */
} Model;

/* A sentence is just a list of HMMs */
struct SentenceStruct {
  int ww;
  Model **ws;
};

typedef struct SentenceStruct *Sentence;

void setGlobalWeights(int a, int b);

void initGlobals(int ll, int uu, RMat *data, int endmode);

Model *allocModel(int ii, int uu);

void copyModel(Model *dst_m, Model *src_m);

Model *cloneModel(Model *src_m);

void uniformModel(Model *m, int ut);

void randomiseModel(Model *m, int ut);

void randomiseModelLinear(Model *m, int ut);

void noiseModel(Model *m, int upper_triangular, Real delta);

void noiseModelLinear(Model *m, int upper_triangular, Real delta);

void smoothModel(Model *m, int upper_triangular, Real eps);

int normalizeModel(Model *m, int upper_triangular);

void defineContFeat(Model *m, int i, Real sigma);

void defineRadialFeat(Model *m, int i);

void defineDiscreteFeat(Model *m, int i);

void displayModel(void *p, Model *m);

void print_model(Model *m);

void freeModel(Model *m);

void zeroModel(Model *m);

void linear2logModel(Model *m);

void log2linearModel(Model *m);

int isZeroModel(Model *m);

Sentence initializeSentence(int ww, Model **ws);

void freeSentence(Sentence s);

Real logLike(Model *m, RMat data);

Real logLike_with_box_scores(Model *m, RMat data, RMat score_mat, int endmode);

int *best_state_sequence(Model *m, RMat data);

void force_init_globals(void);

void compute_posterior
(Model **m, RMat *data, Real *prior, int *c_ls, int ll, int cc, 
 enum training_mode training_mode, 
 /* outputs */ Real *O, Real *like, Real **postpC);

int update
(Model **m, Hmm **tmp_hmm, RMat *data, Real **postpC, Real log_D,
 int *c_ls, int ll, int cc, enum training_mode training_mode, int upper_triangular);

int update_with_box_scores(Model **m, Hmm **tmp_hmm, RMat *data, Real **postpC, Real log_D,
	   int *c_ls, int ll, int cc, enum training_mode training_mode,
			   int upper_triangular, RMat *score_matrices,int endmode);

/* -- Sentence training functions -- */

void decode_box_states(int j, int ii, int nTracks, int *js);

void prune_tracker_states
(Sentence s, int nTracks, IVec *roles, int top_n, int ii, IVec *ped, IVec **ped2);

void prune_xHMM_states(Sentence s, int uu, IVec *pes, IVec **pes2, IVec **pes3);

void compute_crossed_likelihood(IVec ed, IVec es, Sentence s, int nTracks, IVec *roles, 
				int ***phi1, int ****phi2, int ****phi0, int tt, int ii, 
				Real **outL, Real ****xh, M0 *m0);

Real viterbi_tracker_score
(Real ***bps, Real **bss, int nTracks, int ii, int tt, IVec ed, IVec *ed2);

void compute_sentence_alpha
(IVec ed, IVec *ed2, IVec es, IVec *es2, int *final, Sentence s, int ii, int tt, int nTracks, 
 Real **a, Real *b, Real ***bps, Real **bss, Real **outL, Real ***xa, Real **xb, Real **alpha, 
 Real **alpha_p, Real ****xi, M0 *m0);

void compute_sentence_beta
(IVec ed, IVec *ed2, IVec es, IVec *es3, int *final, int ii, int tt, int nTracks, 
 Real **a, Real ***bps, Real **bss, Real **outL, Real **beta, M0 *m0);

void compute_sentence_gamma
(int uu, int tt, Real **alpha, Real **beta, Real **gamma);

void compute_sentence_priors(Sentence *ss, int nn, int tt, Real *priors);

Real compute_sentence_statistics(Sentence s, Hmm *xhmm, int nTracks, IVec *roles,
				 Real **bss, Real ***bps, int ***phi1, int ****phi2, 
				 int ****phi0, int tt, int ii, int final_state,
				 IVec ed, IVec *ed2, IVec es, IVec *es2, IVec *es3,
				 /* Ouput */
				 Real **gamma, Real **alpha, Real **beta, Real **outL, M0 *m0);

Real *sentence_likelihoods_one_video
(Sentence *ss, int nn, int *nTracks, IVec **roles, int ***phi1, int ****phi2, int ****phi0, 
 Real **bss, Real ***bps, int tt, int ii, int top_n, int final_state);

Real sentence_maximum_one
(Sentence s, int nTracks, IVec *roles, int ***phi1, int ****phi2, int ****phi0, 
 Real **bss, Real ***bps, int tt, int ii, int top_n, int final_state, int *box_sequence);

Real viterbi_sentence_tracker
(Sentence s, int nTracks, IVec *roles, int ***phi1, int ****phi2, int ****phi0, Real **bss, 
 Real ***bps, int tt, int ii, int top_n, int sum_or_max, int final_state, int *box_sequence);

Real updateX(Sentence s, Sentence scratch, int nTracks, IVec *roles,
	     int final_state, int ***phi1, int ****phi2, int ****phi0,
	     Real **bss, Real ***bps, int tt, int ii, int top_n);

void HMM_updateXModel(Sentence s, Sentence tmp, Hmm *xhmm, int tt, int ii, Real **bss, Real ***bps, 
		      IVec ed, IVec *ed2, IVec es, IVec *es3, 
		      int nTracks, Real **alpha, Real **beta, Real **outL, Real *gamma0, M0 *m0);

void FFM_updateXModel(Sentence s, Sentence tmp, IVec *roles, Real **gamma, IVec ed, IVec *ed2,
		      IVec es, int nTracks, int ***phi1, int ****phi2, int ****phi0,
		      int tt, int ii, M0 *m0);

void update_objective_derivatives(int nn, Real e, Sentence *scratches, Sentence *tmps, Real *likelihoods);

Real *sentence_derivatives_one_video
(Sentence *ss, Sentence *tmps, int nn, int *nTracks, IVec *roles, int ***phi1, int ****phi2, 
 int ****phi0, Real **bss, Real ***bps, int tt, int ii, int top_n, int final_state);

Real sentence_derivatives_reverse_mode
(Sentence s, Sentence tmp, int nTracks, IVec *roles, int ***phi1, int ****phi2, int ****phi0, Real **bss, 
 Real ***bps, int tt, int ii, int top_n, int final_state);

void removeStates(Model *m, int *xu, Hmm* hmm);

int model_ii(Model *m);

int model_nn(Model *m, int i);

int model_feature_type(Model *m, int i);

Real model_parameter(Model *m, int u, int i, int n);

int model_uu(Model *m);

Hmm *model_hmm(Model *m);

Ffm *model_ffm(Model *m, int i);

Real model_a(Model *m, int u, int v);

Real model_b(Model *m, int u);

char *model_name(Model *m);

int model_n_roles(Model *m);

enum part_of_speech model_pos(Model *m);

void set_model_parameter(Model *m, int u, int i, int n, Real x);

void set_model_a(Model *m, int u, int v, Real x);

void set_model_b(Model *m, int u, Real x);

void set_model_name(Model *m, char * name);

void set_model_n_roles(Model *m, int n);

void set_model_pos(Model *m, enum part_of_speech pos);

char *model_pos_str(enum part_of_speech pos);

void set_min_duration(int new_min_duration);

int get_min_duration();

void switch_duration(int bvalue);

void switch_displacement(int bvalue);

int get_displacement();

void switch_sentence_prior(int bprior);

void set_model_constraint(enum model_constraint new_constraint);

void set_state_constraint(enum state_constraint new_constraint);

int get_duration();

int get_pad_all();

Real single_hmm_run(int **data, int tt, Model *m, Model *scratch, int baum_welch);

void gt_with_derivatives_ml(Model *m, Model *scratch);

#endif

/* Tam V'Nishlam Shevah L'El Borei Olam */
