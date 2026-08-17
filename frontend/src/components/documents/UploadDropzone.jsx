import { useCallback, useMemo, useRef, useState } from 'react';
import { FileUp, Link2, Loader2, Upload, X } from 'lucide-react';
import { Button } from '../common';
import { useSystem } from '../../context/SystemContext';
import { formatBytes } from '../../utils/format';
import cn from '../../utils/cn';

/**
 * Drag-and-drop upload with client-side pre-validation.
 *
 * Files are checked against the backend's advertised extension list and size
 * limit *before* upload, so an obviously-invalid file fails instantly instead of
 * after a slow transfer. The backend re-validates everything regardless — this
 * is a convenience, not the security boundary.
 */
export default function UploadDropzone({ onUpload, onIngestUrl, uploading, progress }) {
  const { settings } = useSystem();
  const [dragging, setDragging] = useState(false);
  const [rejected, setRejected] = useState([]);
  const [urlMode, setUrlMode] = useState(false);
  const [url, setUrl] = useState('');
  const inputRef = useRef(null);

  // Memoised so the identity is stable — `validate` depends on it.
  const allowed = useMemo(
    () =>
      settings?.ingestion?.allowed_extensions || [
        '.pdf', '.docx', '.txt', '.md', '.csv', '.png', '.jpg', '.jpeg',
      ],
    [settings?.ingestion?.allowed_extensions],
  );
  const maxMb = settings?.ingestion?.max_upload_mb || 50;

  const validate = useCallback(
    (files) => {
      const accepted = [];
      const failures = [];

      Array.from(files).forEach((file) => {
        const extension = file.name.includes('.')
          ? `.${file.name.split('.').pop().toLowerCase()}`
          : '';
        if (!allowed.includes(extension)) {
          failures.push({
            name: file.name,
            reason: `'${extension || 'unknown'}' is not a supported type.`,
          });
          return;
        }
        if (file.size > maxMb * 1024 * 1024) {
          failures.push({
            name: file.name,
            reason: `${formatBytes(file.size)} exceeds the ${maxMb} MB limit.`,
          });
          return;
        }
        if (file.size === 0) {
          failures.push({ name: file.name, reason: 'The file is empty.' });
          return;
        }
        accepted.push(file);
      });

      setRejected(failures);
      return accepted;
    },
    [allowed, maxMb],
  );

  const handleFiles = useCallback(
    (files) => {
      const accepted = validate(files);
      if (accepted.length) onUpload(accepted);
    },
    [validate, onUpload],
  );

  const onDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    if (event.dataTransfer.files?.length) handleFiles(event.dataTransfer.files);
  };

  const submitUrl = () => {
    if (!url.trim()) return;
    onIngestUrl(url.trim());
    setUrl('');
    setUrlMode(false);
  };

  return (
    <div>
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={cn(
          'rounded-xl border-2 border-dashed p-6 text-center transition sm:p-8',
          dragging ? 'border-brand-400 bg-brand-50/60' : 'border-ink-200 bg-white hover:border-ink-300',
          uploading && 'pointer-events-none opacity-70',
        )}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={allowed.join(',')}
          className="sr-only"
          onChange={(event) => {
            if (event.target.files?.length) handleFiles(event.target.files);
            event.target.value = '';
          }}
          aria-label="Choose files to upload"
        />

        <span
          className={cn(
            'mx-auto flex h-12 w-12 items-center justify-center rounded-full',
            dragging ? 'bg-brand-100 text-brand-600' : 'bg-ink-100 text-ink-400',
          )}
        >
          {uploading ? (
            <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
          ) : (
            <Upload className="h-5 w-5" aria-hidden="true" />
          )}
        </span>

        {uploading ? (
          <>
            <p className="mt-3 text-sm font-medium text-ink-900">Uploading…</p>
            <div className="mx-auto mt-3 h-1.5 w-full max-w-xs overflow-hidden rounded-full bg-ink-100">
              <div
                className="h-full rounded-full bg-brand-600 transition-[width] duration-200"
                style={{ width: `${progress || 0}%` }}
              />
            </div>
            <p className="mt-1.5 text-xs text-ink-500">{progress || 0}%</p>
          </>
        ) : (
          <>
            <p className="mt-3 text-sm font-medium text-ink-900">
              Drop documents here, or{' '}
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="text-brand-600 underline underline-offset-2 hover:text-brand-700"
              >
                browse
              </button>
            </p>
            <p className="mt-1 text-xs text-ink-500">
              {allowed.join(' · ')} — up to {maxMb} MB each
            </p>

            <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
              <Button variant="secondary" size="sm" icon={FileUp} onClick={() => inputRef.current?.click()}>
                Select files
              </Button>
              <Button variant="ghost" size="sm" icon={Link2} onClick={() => setUrlMode((open) => !open)}>
                Ingest a URL
              </Button>
            </div>
          </>
        )}
      </div>

      {/* URL ingestion */}
      {urlMode && !uploading ? (
        <div className="mt-3 flex flex-col gap-2 rounded-xl border border-ink-200 bg-white p-3 sm:flex-row">
          <label htmlFor="ingest-url" className="sr-only">
            Page URL
          </label>
          <input
            id="ingest-url"
            type="url"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            onKeyDown={(event) => event.key === 'Enter' && submitUrl()}
            placeholder="https://example.com/paper"
            className="min-w-0 flex-1 rounded-lg border border-ink-200 px-3 py-2 text-sm focus:border-brand-400 focus:outline-none focus:ring-1 focus:ring-brand-400"
          />
          <Button size="sm" onClick={submitUrl} disabled={!url.trim()}>
            Fetch and index
          </Button>
        </div>
      ) : null}

      {/* Client-side rejections */}
      {rejected.length ? (
        <div className="mt-3 rounded-xl border border-rose-200 bg-rose-50 p-3">
          <div className="flex items-start justify-between gap-2">
            <p className="text-xs font-semibold text-rose-900">
              {rejected.length} file(s) were not accepted
            </p>
            <button
              type="button"
              onClick={() => setRejected([])}
              className="shrink-0 rounded p-0.5 text-rose-400 hover:bg-rose-100 hover:text-rose-700"
              aria-label="Dismiss"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
          <ul className="mt-1.5 space-y-0.5">
            {rejected.map((item) => (
              <li key={item.name} className="text-xs text-rose-700">
                <span className="font-medium">{item.name}</span> — {item.reason}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
