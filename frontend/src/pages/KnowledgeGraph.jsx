import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { ReactFlowProvider } from 'reactflow';
import {
  FileText,
  GitBranch,
  Network,
  RefreshCw,
  Route,
  Search,
  Share2,
  SlidersHorizontal,
  X,
} from 'lucide-react';
import {
  Button,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorState,
  InfoBanner,
  PageHeader,
  StatTile,
} from '../components/common';
import GraphCanvas from '../components/graph/GraphCanvas';
import { useApi } from '../hooks/useApi';
import { useDebounce } from '../hooks/useDebounce';
import { graphService } from '../services';
import { ENTITY_TYPE_COLORS } from '../utils/constants';
import { formatNumber, formatRatio } from '../utils/format';
import cn from '../utils/cn';

export default function KnowledgeGraph() {
  const [searchParams, setSearchParams] = useSearchParams();
  const documentFilter = searchParams.get('document') || '';

  const [search, setSearch] = useState('');
  const [selectedId, setSelectedId] = useState(null);
  const [limit, setLimit] = useState(150);
  const [pathTarget, setPathTarget] = useState('');
  const [paths, setPaths] = useState(null);
  const [pathLoading, setPathLoading] = useState(false);

  const debouncedSearch = useDebounce(search, 300);

  const { data: graph, error, loading, refetch } = useApi(
    () => graphService.export({ limit, documentId: documentFilter || undefined }),
    [limit, documentFilter],
  );
  const { data: stats } = useApi(() => graphService.stats(), []);

  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];

  const selectedNode = useMemo(
    () => nodes.find((node) => node.id === selectedId) || null,
    [nodes, selectedId],
  );

  const selectedEdges = useMemo(
    () =>
      selectedId
        ? edges.filter((edge) => edge.source === selectedId || edge.target === selectedId)
        : [],
    [edges, selectedId],
  );

  const nameById = useMemo(
    () => new Map(nodes.map((node) => [node.id, node.name])),
    [nodes],
  );

  // Clear a stale path result whenever the anchor entity changes.
  useEffect(() => {
    setPaths(null);
    setPathTarget('');
  }, [selectedId]);

  const findPaths = async () => {
    if (!selectedNode || !pathTarget.trim()) return;
    setPathLoading(true);
    try {
      const result = await graphService.paths(selectedNode.name, pathTarget.trim(), 4);
      setPaths(result);
    } catch {
      setPaths({ found: false, paths: [] });
    } finally {
      setPathLoading(false);
    }
  };

  const searchMatches = useMemo(() => {
    const term = debouncedSearch.trim().toLowerCase();
    if (!term) return [];
    return nodes.filter((node) => node.name.toLowerCase().includes(term)).slice(0, 8);
  }, [debouncedSearch, nodes]);

  return (
    <>
      <PageHeader
        title="Knowledge Graph"
        description="Entities and typed relations extracted during ingestion. This is what Graph RAG traverses to answer relationship and multi-hop questions."
        action={
          <Button variant="secondary" size="sm" icon={RefreshCw} onClick={refetch} loading={loading}>
            Refresh
          </Button>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile label="Entities" value={formatNumber(stats?.entities)} icon={Network} tone="emerald" />
        <StatTile label="Relations" value={formatNumber(stats?.relations)} icon={GitBranch} tone="brand" />
        <StatTile
          label="Entity types"
          value={formatNumber(Object.keys(stats?.entity_types || {}).length)}
          icon={Share2}
          tone="violet"
        />
        <StatTile
          label="Backend"
          value={stats?.backend === 'neo4j' ? 'Neo4j' : 'NetworkX'}
          hint={stats?.backend === 'neo4j' ? 'Cypher traversal' : 'Embedded graph store'}
          icon={SlidersHorizontal}
          tone="ink"
        />
      </div>

      {documentFilter ? (
        <InfoBanner
          variant="info"
          className="mt-4"
          icon={FileText}
          action={
            <Button variant="ghost" size="xs" icon={X} onClick={() => setSearchParams({})}>
              Clear
            </Button>
          }
        >
          Showing only entities extracted from one document.
        </InfoBanner>
      ) : null}

      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        {/* --------------------------------------------------------- canvas */}
        <Card className="overflow-hidden lg:col-span-2">
          <div className="flex flex-col gap-2 border-b border-ink-100 p-3 sm:flex-row sm:items-center">
            <div className="relative min-w-0 flex-1">
              <Search
                className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-400"
                aria-hidden="true"
              />
              <input
                type="search"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search entities…"
                className="w-full rounded-lg border border-ink-200 py-1.5 pl-9 pr-3 text-sm placeholder:text-ink-400 focus:border-brand-400 focus:outline-none focus:ring-1 focus:ring-brand-400"
                aria-label="Search entities"
              />

              {searchMatches.length ? (
                <ul className="absolute left-0 right-0 top-full z-20 mt-1 max-h-56 overflow-y-auto rounded-lg border border-ink-200 bg-white py-1 shadow-panel">
                  {searchMatches.map((node) => (
                    <li key={node.id}>
                      <button
                        type="button"
                        onClick={() => {
                          setSelectedId(node.id);
                          setSearch('');
                        }}
                        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs transition hover:bg-ink-50"
                      >
                        <span
                          className="h-1.5 w-1.5 shrink-0 rounded-full"
                          style={{
                            backgroundColor:
                              ENTITY_TYPE_COLORS[node.type] || ENTITY_TYPE_COLORS.CONCEPT,
                          }}
                          aria-hidden="true"
                        />
                        <span className="min-w-0 flex-1 truncate text-ink-800">{node.name}</span>
                        <span className="shrink-0 text-[10px] text-ink-400">{node.degree} links</span>
                      </button>
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>

            <select
              value={limit}
              onChange={(event) => setLimit(Number(event.target.value))}
              className="rounded-lg border border-ink-200 px-2.5 py-1.5 text-xs text-ink-700 focus:border-brand-400 focus:outline-none"
              aria-label="Maximum nodes"
            >
              {[50, 100, 150, 250, 400].map((value) => (
                <option key={value} value={value}>
                  {value} nodes
                </option>
              ))}
            </select>
          </div>

          <div className="h-[540px] w-full bg-ink-50/40">
            {error ? (
              <ErrorState error={error} onRetry={refetch} />
            ) : !nodes.length && !loading ? (
              <EmptyState
                icon={Network}
                title="The knowledge graph is empty"
                description="Entities and relations are extracted during ingestion and require a cloud LLM provider. Upload a document with a Gemini or Groq key configured to populate the graph."
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
            ) : (
              <ReactFlowProvider>
                <GraphCanvas
                  nodes={nodes}
                  edges={edges}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                  searchTerm={debouncedSearch}
                />
              </ReactFlowProvider>
            )}
          </div>

          {graph?.truncated ? (
            <p className="border-t border-ink-100 px-4 py-2 text-[11px] text-ink-500">
              Showing the {nodes.length} most-connected entities. Increase the node limit to see more.
            </p>
          ) : null}
        </Card>

        {/* -------------------------------------------------------- inspector */}
        <div className="space-y-4">
          {selectedNode ? (
            <>
              <Card>
                <CardHeader
                  title={selectedNode.name}
                  description={`${selectedNode.type} · ${selectedEdges.length} connection(s)`}
                  action={
                    <button
                      type="button"
                      onClick={() => setSelectedId(null)}
                      className="rounded p-1 text-ink-400 hover:bg-ink-100 hover:text-ink-700"
                      aria-label="Clear selection"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  }
                />
                <CardBody>
                  {selectedNode.description ? (
                    <p className="mb-3 text-xs leading-relaxed text-ink-600">
                      {selectedNode.description}
                    </p>
                  ) : null}

                  {selectedEdges.length ? (
                    <ul className="space-y-1.5">
                      {selectedEdges.map((edge) => {
                        const outgoing = edge.source === selectedId;
                        const otherId = outgoing ? edge.target : edge.source;
                        return (
                          <li key={edge.id} className="rounded-lg bg-ink-50 p-2">
                            <p className="flex flex-wrap items-center gap-1 text-[11px]">
                              {!outgoing ? (
                                <button
                                  type="button"
                                  onClick={() => setSelectedId(otherId)}
                                  className="font-medium text-brand-600 hover:underline"
                                >
                                  {nameById.get(otherId) || otherId}
                                </button>
                              ) : (
                                <span className="font-medium text-ink-900">{selectedNode.name}</span>
                              )}
                              <span className="rounded bg-white px-1.5 py-0.5 font-mono text-[10px] text-ink-600">
                                {edge.type?.replace(/_/g, ' ').toLowerCase()}
                              </span>
                              {outgoing ? (
                                <button
                                  type="button"
                                  onClick={() => setSelectedId(otherId)}
                                  className="font-medium text-brand-600 hover:underline"
                                >
                                  {nameById.get(otherId) || otherId}
                                </button>
                              ) : (
                                <span className="font-medium text-ink-900">{selectedNode.name}</span>
                              )}
                            </p>
                            {edge.context ? (
                              <p className="mt-1 line-clamp-2 text-[10px] italic leading-relaxed text-ink-500">
                                “{edge.context}”
                              </p>
                            ) : null}
                            <p className="mt-0.5 text-[10px] text-ink-400">
                              confidence {formatRatio(edge.confidence, 2)}
                            </p>
                          </li>
                        );
                      })}
                    </ul>
                  ) : (
                    <p className="text-xs text-ink-400">
                      No relations were extracted for this entity — it appears only as a mention.
                    </p>
                  )}

                  {selectedNode.documents?.length ? (
                    <div className="mt-3 border-t border-ink-100 pt-3">
                      <p className="mb-1.5 text-[11px] font-medium text-ink-500">Appears in</p>
                      <ul className="space-y-1">
                        {selectedNode.documents.map((docId) => (
                          <li key={docId}>
                            <Link
                              to={`/knowledge-base/${docId}`}
                              className="flex items-center gap-1.5 truncate text-[11px] text-brand-600 hover:text-brand-700"
                            >
                              <FileText className="h-3 w-3 shrink-0" aria-hidden="true" />
                              <span className="truncate font-mono">{docId.slice(0, 12)}…</span>
                            </Link>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                </CardBody>
              </Card>

              {/* Path finder — the multi-hop traversal, exposed directly. */}
              <Card>
                <CardHeader title="Find a path" description="How does this entity connect to another?" icon={Route} />
                <CardBody>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={pathTarget}
                      onChange={(event) => setPathTarget(event.target.value)}
                      onKeyDown={(event) => event.key === 'Enter' && findPaths()}
                      placeholder="Target entity…"
                      className="min-w-0 flex-1 rounded-lg border border-ink-200 px-2.5 py-1.5 text-xs focus:border-brand-400 focus:outline-none focus:ring-1 focus:ring-brand-400"
                    />
                    <Button size="xs" onClick={findPaths} loading={pathLoading} disabled={!pathTarget.trim()}>
                      Trace
                    </Button>
                  </div>

                  {paths ? (
                    paths.found ? (
                      <ul className="mt-3 space-y-2">
                        {paths.paths.slice(0, 5).map((path, index) => (
                          <li key={index} className="rounded-lg bg-emerald-50 p-2">
                            <p className="font-mono text-[10px] leading-relaxed text-emerald-800">
                              {path.description}
                            </p>
                            <p className="mt-0.5 text-[10px] text-emerald-600">
                              {path.relations.length} hop(s) · score {formatRatio(path.score, 2)}
                            </p>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="mt-3 text-[11px] text-ink-500">
                        No path was found between these entities within 4 hops.
                      </p>
                    )
                  ) : null}
                </CardBody>
              </Card>
            </>
          ) : (
            <>
              <Card>
                <CardHeader title="Entity types" />
                <CardBody>
                  {Object.keys(stats?.entity_types || {}).length ? (
                    <ul className="space-y-1.5">
                      {Object.entries(stats.entity_types).map(([type, count]) => (
                        <li key={type} className="flex items-center gap-2">
                          <span
                            className="h-2 w-2 shrink-0 rounded-full"
                            style={{
                              backgroundColor: ENTITY_TYPE_COLORS[type] || ENTITY_TYPE_COLORS.CONCEPT,
                            }}
                            aria-hidden="true"
                          />
                          <span className="flex-1 truncate text-xs text-ink-700">{type}</span>
                          <span className="shrink-0 text-[11px] tabular-nums text-ink-500">{count}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-xs text-ink-400">No entities extracted yet.</p>
                  )}
                </CardBody>
              </Card>

              <Card>
                <CardHeader title="Most connected" description="Hub entities in the corpus." />
                <CardBody>
                  {stats?.top_entities?.length ? (
                    <ul className="space-y-1">
                      {stats.top_entities.map((entity) => {
                        const node = nodes.find((item) => item.name === entity.name);
                        return (
                          <li key={entity.name}>
                            <button
                              type="button"
                              disabled={!node}
                              onClick={() => node && setSelectedId(node.id)}
                              className={cn(
                                'flex w-full items-center gap-2 rounded px-1.5 py-1 text-left transition',
                                node ? 'hover:bg-ink-50' : 'cursor-default opacity-60',
                              )}
                            >
                              <span className="min-w-0 flex-1 truncate text-xs text-ink-700">
                                {entity.name}
                              </span>
                              <span className="shrink-0 text-[11px] tabular-nums text-ink-400">
                                {entity.degree}
                              </span>
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  ) : (
                    <p className="text-xs text-ink-400">No entities yet.</p>
                  )}
                </CardBody>
              </Card>

              <Card>
                <CardBody>
                  <p className="text-[11px] leading-relaxed text-ink-500">
                    Select a node to see its relations, the passages each relation was extracted
                    from, and to trace a path to another entity — the same traversal Graph RAG runs
                    when answering a relationship question.
                  </p>
                </CardBody>
              </Card>
            </>
          )}
        </div>
      </div>
    </>
  );
}
