import { Link } from 'react-router-dom';

export default function NotFound({ message = 'There is no page at this address.' }) {
  return (
    <section className="not-found">
      <h1>Not found</h1>
      <p>{message}</p>
      <p>
        <Link to="/">Back to the component graph</Link> — every component is listed there.
      </p>
    </section>
  );
}
