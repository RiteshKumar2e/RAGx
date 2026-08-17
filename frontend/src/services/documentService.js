import { api, del, get, post, resourceUrl } from './api';

/** Knowledge-base operations: upload, listing, inspection, deletion, reindexing. */
export const documentService = {
  list({ page = 1, pageSize = 20, status, fileType, search } = {}) {
    return get('/documents', {
      params: {
        page,
        page_size: pageSize,
        status: status || undefined,
        file_type: fileType || undefined,
        search: search || undefined,
      },
    });
  },

  stats() {
    return get('/documents/stats');
  },

  detail(documentId, chunkLimit = 60) {
    return get(`/documents/${documentId}`, { params: { chunk_limit: chunkLimit } });
  },

  /**
   * Upload files with progress reporting.
   * `onProgress` receives 0–100. The backend validates every file and returns
   * both accepted and rejected entries, so a bad file never blocks the rest.
   */
  upload(files, { onProgress, signal } = {}) {
    const form = new FormData();
    Array.from(files).forEach((file) => form.append('files', file));

    return api
      .post('/documents/upload', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 0, // uploads are bounded by size, not by the API timeout
        signal,
        onUploadProgress: (event) => {
          if (onProgress && event.total) {
            onProgress(Math.round((event.loaded * 100) / event.total));
          }
        },
      })
      .then((response) => response.data);
  },

  ingestUrl(url, title) {
    return post('/documents/ingest-url', { url, title: title || undefined });
  },

  reindex(documentId) {
    return post(`/documents/${documentId}/reindex`);
  },

  remove(documentId) {
    return del(`/documents/${documentId}`);
  },

  rebuildIndexes() {
    return post('/documents/rebuild-indexes');
  },

  fileUrl(documentId) {
    return resourceUrl(`/documents/${documentId}/file`);
  },
};

export default documentService;
