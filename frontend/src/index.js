import React from 'react';
import ReactDOM from 'react-dom';
import Controls from './Controls';
import './index.css';
import Viz from './Viz';
import Camera from './Camera';

ReactDOM.render(
  <React.StrictMode>
    {/* <Camera />
    <Controls /> */}
    <Viz />
  </React.StrictMode>,
  document.getElementById('root')
);