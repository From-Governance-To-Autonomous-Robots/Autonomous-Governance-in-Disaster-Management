import React from 'react';
import ReactDOM from 'react-dom';
import './styles/global.css'; // Global CSS styles
import App from './App'; // Main App component
import { BrowserRouter as Router } from 'react-router-dom';

ReactDOM.render(
  <React.StrictMode>
    <Router>
      <App />
    </Router>
  </React.StrictMode>,
  document.getElementById('root')
);
