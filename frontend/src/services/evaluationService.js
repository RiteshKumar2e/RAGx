import { del, get, post } from './api';

/** Benchmark inspection, experiment runs and strategy comparison. */
export const evaluationService = {
  benchmark(dataset = 'ragx_benchmark') {
    return get('/evaluation/benchmark', { params: { dataset } });
  },

  datasets() {
    return get('/evaluation/datasets');
  },

  /**
   * Start one run per strategy over the same questions.
   * Runs execute in the background; poll `runs()` for progress.
   */
  run({ strategies, dataset = 'ragx_benchmark', categories, limit, k = 8, judgeGeneration = true, name, notes }) {
    return post('/evaluation/run', {
      strategies,
      dataset,
      categories: categories?.length ? categories : undefined,
      limit: limit || undefined,
      k,
      judge_generation: judgeGeneration,
      name: name || undefined,
      notes: notes || undefined,
    });
  },

  runs({ limit = 50, strategy } = {}) {
    return get('/evaluation/runs', { params: { limit, strategy: strategy || undefined } });
  },

  run_(runId) {
    return get(`/evaluation/runs/${runId}`);
  },

  results(runId, limit = 200) {
    return get(`/evaluation/runs/${runId}/results`, { params: { limit } });
  },

  /** Latest completed run per strategy, side by side. */
  comparison() {
    return get('/evaluation/comparison');
  },

  deleteRun(runId) {
    return del(`/evaluation/runs/${runId}`);
  },
};

export default evaluationService;
