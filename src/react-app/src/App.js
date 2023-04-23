// App.js
// import './App.css';
import React from 'react';
import { Route, Routes } from 'react-router-dom';

import MyIndex from './MyIndex';
import Dataload from './Dataload';
import Upload from './Upload';
import Login from './Login';
import Firewall from "./Firewall";
import Staging from "./Staging";

function App() {
  return (
    <div className="App">
      <header>
      </header>
      <Routes>
        <Route path="/Dataload" element={<Dataload />} />
        <Route path="/Dataload" element={<Firewall />} />
        <Route path="/Dataload" element={<Staging />} />
        <Route path="/Upload" element={<Upload />} />
        <Route path="/Login" element={<Login />} />
        <Route path="/" element={<MyIndex />} />
      </Routes>
    </div>
  );
}

export default App;