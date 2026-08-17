import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Library, MessageSquareText, PanelRightClose, PanelRightOpen, Plus, Sparkles } from 'lucide-react';
import MessageList from '../components/chat/MessageList';
import QueryInput from '../components/chat/QueryInput';
import EvidencePanel from '../components/evidence/EvidencePanel';
import { Button, EmptyState, ErrorState } from '../components/common';
import { useSystem } from '../context/SystemContext';
import { useToast } from '../context/ToastContext';
import { documentService, queryService } from '../services';
import cn from '../utils/cn';

const SUGGESTIONS = [
  { text: 'What is the main contribution described in the abstract?', hint: 'simple → Naive RAG' },
  { text: 'Which dataset and metric are reported, and what score was achieved?', hint: 'exact terms → Hybrid RAG' },
  { text: 'How does the proposed method relate to the prior work it builds on?', hint: 'relationships → Graph RAG' },
  { text: 'What does Figure 1 show, and what trend is visible in it?', hint: 'figures → Multimodal RAG' },
  {
    text: 'What are the main limitations, what evidence supports each, and what would fix them?',
    hint: 'multi-hop → Agentic RAG',
  },
];

let messageCounter = 0;
const nextId = () => `m${++messageCounter}`;

export default function ResearchAssistant() {
  const { usingDevEmbedder } = useSystem();
  const toast = useToast();

  const [messages, setMessages] = useState([]);
  const [evidence, setEvidence] = useState([]);
  const [activeChunkId, setActiveChunkId] = useState(null);
  const [conversationId, setConversationId] = useState(null);
  const [pending, setPending] = useState(false);
  const [pendingStage, setPendingStage] = useState('analyzing');
  const [partialAnswer, setPartialAnswer] = useState('');
  const [error, setError] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [showEvidence, setShowEvidence] = useState(true);

  const abortRef = useRef(null);

  // Ready documents power the "restrict to documents" picker and the empty state.
  useEffect(() => {
    documentService
      .list({ pageSize: 100, status: 'ready' })
      .then((payload) => setDocuments(payload.items || []))
      .catch(() => setDocuments([]));
  }, []);

  const hasDocuments = documents.length > 0;

  const focusEvidence = useCallback((chunkId) => {
    setActiveChunkId(chunkId);
    setShowEvidence(true);
    // Let the panel render before scrolling to the card.
    requestAnimationFrame(() => {
      document
        .getElementById(`evidence-${chunkId}`)
        ?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  }, []);

  const runQuery = useCallback(
    async ({ question, strategies, documentIds, topK }) => {
      setError(null);
      setPending(true);
      setPendingStage('analyzing');
      setPartialAnswer('');
      setMessages((current) => [...current, { id: nextId(), role: 'user', content: question }]);

      const controller = new AbortController();
      abortRef.current = controller;

      await queryService.stream(
        { question, conversationId, strategies, documentIds, topK },
        {
          signal: controller.signal,
          onStatus: (status) => {
            setPendingStage(status.stage === 'retrieved' ? 'generating' : status.stage);
          },
          onToken: (text) => setPartialAnswer((current) => current + text),
          onDone: (payload) => {
            setMessages((current) => [
              ...current,
              { id: nextId(), role: 'assistant', ...payload },
            ]);
            setEvidence(payload.evidence || []);
            setConversationId(payload.conversation_id);
            setPending(false);
            setPartialAnswer('');
            abortRef.current = null;

            if (payload.abstained) {
              toast.warning(
                'RAGX withheld the answer — the retrieved evidence did not support one.',
                { title: 'Insufficient evidence' },
              );
            }
          },
          onError: (payload) => {
            setPending(false);
            setPartialAnswer('');
            abortRef.current = null;
            setError({ message: payload?.message || 'The query failed.' });
          },
        },
      );
    },
    [conversationId, toast],
  );

  const cancel = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setPending(false);
    setPartialAnswer('');
    toast.info('Query cancelled.');
  };

  const newThread = () => {
    abortRef.current?.abort();
    setMessages([]);
    setEvidence([]);
    setActiveChunkId(null);
    setConversationId(null);
    setPending(false);
    setPartialAnswer('');
    setError(null);
  };

  return (
    <div className="flex h-[calc(100vh-7rem)] flex-col gap-4 lg:flex-row">
      {/* ----------------------------------------------------- conversation */}
      <section
        className={cn(
          'flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-ink-200 bg-white',
          showEvidence ? 'lg:w-[58%]' : 'lg:w-full',
        )}
      >
        <header className="flex shrink-0 items-center justify-between gap-3 border-b border-ink-100 px-4 py-3">
          <div className="flex min-w-0 items-center gap-2">
            <MessageSquareText className="h-4 w-4 shrink-0 text-brand-600" aria-hidden="true" />
            <h2 className="truncate text-sm font-semibold text-ink-900">Research Assistant</h2>
            {messages.length ? (
              <span className="shrink-0 text-[11px] text-ink-400">
                {messages.filter((m) => m.role === 'user').length} question(s)
              </span>
            ) : null}
          </div>

          <div className="flex shrink-0 items-center gap-1">
            {messages.length ? (
              <Button variant="ghost" size="xs" icon={Plus} onClick={newThread}>
                New thread
              </Button>
            ) : null}
            <button
              type="button"
              onClick={() => setShowEvidence((open) => !open)}
              className="hidden rounded-lg p-1.5 text-ink-400 transition hover:bg-ink-100 hover:text-ink-700 lg:block"
              aria-label={showEvidence ? 'Hide evidence panel' : 'Show evidence panel'}
              title={showEvidence ? 'Hide evidence panel' : 'Show evidence panel'}
            >
              {showEvidence ? (
                <PanelRightClose className="h-4 w-4" />
              ) : (
                <PanelRightOpen className="h-4 w-4" />
              )}
            </button>
          </div>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {error ? (
            <ErrorState error={error} onRetry={() => setError(null)} compact className="mb-4" />
          ) : null}

          {!messages.length && !pending ? (
            hasDocuments ? (
              <div className="flex h-full flex-col items-center justify-center px-4 text-center">
                <span className="rounded-full bg-brand-50 p-3 text-brand-600">
                  <Sparkles className="h-6 w-6" aria-hidden="true" />
                </span>
                <h3 className="mt-4 text-sm font-semibold text-ink-900">
                  Ask a question about your indexed documents
                </h3>
                <p className="mt-1 max-w-md text-sm text-ink-500">
                  RAGX analyses the question first, then selects the retrieval strategies that can
                  actually answer it. Try a simple lookup and a multi-hop question to see the routing
                  change.
                </p>

                <div className="mt-6 w-full max-w-2xl space-y-2">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion.text}
                      type="button"
                      disabled={pending}
                      onClick={() => runQuery({ question: suggestion.text })}
                      className="flex w-full items-center justify-between gap-3 rounded-lg border border-ink-200 bg-white px-3.5 py-2.5 text-left transition hover:border-brand-300 hover:bg-brand-50/40 disabled:opacity-50"
                    >
                      <span className="min-w-0 truncate text-sm text-ink-700">{suggestion.text}</span>
                      <span className="shrink-0 font-mono text-[10px] text-ink-400">
                        {suggestion.hint}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <EmptyState
                icon={Library}
                title="No documents indexed yet"
                description="RAGX answers strictly from the documents you upload. Add a PDF, DOCX, CSV, text file or image to get started."
                action={
                  <Link
                    to="/knowledge-base"
                    className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-3.5 py-2 text-sm font-medium text-white transition hover:bg-brand-700"
                  >
                    Go to Knowledge Base
                  </Link>
                }
                className="h-full"
              />
            )
          ) : (
            <MessageList
              messages={messages}
              pending={pending}
              pendingStage={pendingStage}
              partialAnswer={partialAnswer}
              onEvidenceFocus={focusEvidence}
            />
          )}
        </div>

        <QueryInput
          onSubmit={runQuery}
          onCancel={cancel}
          busy={pending}
          disabled={!hasDocuments}
          disabledReason="Upload a document before asking a question"
          documents={documents}
        />
      </section>

      {/* --------------------------------------------------------- evidence */}
      {showEvidence ? (
        <aside className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-ink-200 bg-white lg:w-[42%] lg:max-w-md">
          {usingDevEmbedder ? (
            <p className="shrink-0 border-b border-amber-200 bg-amber-50 px-4 py-2 text-[11px] leading-relaxed text-amber-800">
              Development embedder active — retrieval matches text lexically, not semantically.
            </p>
          ) : null}
          <EvidencePanel
            evidence={evidence}
            activeChunkId={activeChunkId}
            onSelect={setActiveChunkId}
            className="min-h-0 flex-1"
          />
        </aside>
      ) : null}
    </div>
  );
}
