import { useEffect } from 'react';

const SITE_NAME = 'RAGX';

/**
 * Sets the browser tab title for the current page. React Router does not touch
 * document.title on navigation, so without this every route shows the title
 * from index.html.
 *
 * Pass `exact: true` to use `title` verbatim instead of appending the site name.
 */
export default function useDocumentTitle(title, { exact = false } = {}) {
  useEffect(() => {
    if (!title) {
      document.title = SITE_NAME;
    } else {
      document.title = exact ? title : `${title} — ${SITE_NAME}`;
    }
  }, [title, exact]);
}
