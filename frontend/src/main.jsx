import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    {/*
      Opt in to the React Router v7 behaviours now, while still on v6:
      - v7_startTransition:  route state updates wrapped in React.startTransition
      - v7_relativeSplatPath: corrected relative resolution inside splat routes
      Enabling them early silences the deprecation warnings and means the v7
      upgrade is a version bump rather than a behavioural change.
    */}
    <BrowserRouter
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
    >
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
