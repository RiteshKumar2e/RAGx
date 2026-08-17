import { Fragment, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const MARKER = /(\[\d{1,2}\])/g;

/**
 * Renders an answer as Markdown with clickable `[n]` citation markers.
 *
 * The markers are turned into buttons at the text-node level, so they work
 * inside paragraphs, list items and table cells alike without breaking the
 * surrounding Markdown structure. A marker that does not resolve to a real
 * evidence block is rendered as plain text rather than a dead link.
 */
function withCitations(children, citationsByMarker, onCitationClick) {
  return Array.isArray(children)
    ? children.map((child, index) => (
        <Fragment key={index}>{transform(child, citationsByMarker, onCitationClick)}</Fragment>
      ))
    : transform(children, citationsByMarker, onCitationClick);
}

function transform(node, citationsByMarker, onCitationClick) {
  if (typeof node !== 'string') return node;
  if (!MARKER.test(node)) return node;

  MARKER.lastIndex = 0;
  return node.split(MARKER).map((part, index) => {
    const match = /^\[(\d{1,2})\]$/.exec(part);
    if (!match) return <Fragment key={index}>{part}</Fragment>;

    const marker = Number(match[1]);
    const citation = citationsByMarker.get(marker);
    if (!citation) return <Fragment key={index}>{part}</Fragment>;

    const label = [citation.document_name, citation.page ? `p.${citation.page}` : null, citation.section]
      .filter(Boolean)
      .join(' · ');

    return (
      <button
        key={index}
        type="button"
        className="citation-marker"
        onClick={() => onCitationClick?.(citation)}
        title={label}
        aria-label={`Citation ${marker}: ${label}`}
      >
        {marker}
      </button>
    );
  });
}

export default function MarkdownAnswer({ content, citations = [], onCitationClick }) {
  const citationsByMarker = useMemo(
    () => new Map(citations.map((citation) => [citation.marker, citation])),
    [citations],
  );

  const components = useMemo(
    () => ({
      p: ({ children }) => <p>{withCitations(children, citationsByMarker, onCitationClick)}</p>,
      li: ({ children }) => <li>{withCitations(children, citationsByMarker, onCitationClick)}</li>,
      td: ({ children }) => <td>{withCitations(children, citationsByMarker, onCitationClick)}</td>,
      th: ({ children }) => <th>{children}</th>,
      strong: ({ children }) => (
        <strong>{withCitations(children, citationsByMarker, onCitationClick)}</strong>
      ),
      em: ({ children }) => <em>{withCitations(children, citationsByMarker, onCitationClick)}</em>,
      // Wide tables scroll inside their own container, never the page.
      table: ({ children }) => (
        <div className="table-scroll">
          <table>{children}</table>
        </div>
      ),
      a: ({ href, children }) => (
        <a href={href} target="_blank" rel="noopener noreferrer">
          {children}
        </a>
      ),
      code: ({ inline, className, children, ...props }) =>
        inline ? (
          <code className={className} {...props}>
            {children}
          </code>
        ) : (
          <code className={className} {...props}>
            {children}
          </code>
        ),
    }),
    [citationsByMarker, onCitationClick],
  );

  return (
    <div className="prose-ragx">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content || ''}
      </ReactMarkdown>
    </div>
  );
}
