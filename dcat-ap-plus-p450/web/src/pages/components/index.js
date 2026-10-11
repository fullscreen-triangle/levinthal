// Dedicated component pages, keyed by component id from components.json.
//
// Every id renders GenericComponentPage unless it has an entry here. To give a
// component its own page, write it (e.g. ./DomCycle.jsx) and register it:
//
//   import DomCycle from './DomCycle.jsx';
//   export const dedicatedPages = { 'dom.cycle': DomCycle };
//
// A dedicated page receives the same props as the generic one:
//   { node, layer, phase, status, incoming, outgoing }
// and may render <GenericComponentPage {...props} /> inside itself to keep
// the standard header and edge lists.
export const dedicatedPages = {};
