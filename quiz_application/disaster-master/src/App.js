import React from 'react';
import { Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage';
import QuizPage from './pages/QuizPage';
import './styles/global.css';
import { AuthProvider } from './context/AuthContext';
import ResultPage from './pages/ResultPage';
import DummyQuizPage from './pages/DummyQuiz';

const App = () => {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/" element={<HomePage/>} />
        <Route path="/training/victim/checkaid" element={<QuizPage task={"info"} phase={"train"}/>} />
        <Route path="/training/victim/typeofaid" element={<QuizPage task={"human"} phase={"train"}/>} />
        <Route path="/training/victim/typeofdamage" element={<QuizPage task={"damage"} phase={"train"}/>} />
        <Route path="/training/satellite/typeofdamage" element={<QuizPage task={"satellite"} phase={"train"}/>} />
        <Route path="/training/drone/typeofdamage" element={<QuizPage task={"drone-damage"} phase={"train"}/>} />
        <Route path="/validation/victim/checkaid" element={<QuizPage task={"info"} phase={"val"}/>} />
        <Route path="/validation/victim/typeofaid" element={<QuizPage task={"human"} phase={"val"}/>} />
        <Route path="/validation/victim/typeofdamage" element={<QuizPage task={"damage"} phase={"val"}/>} />
        <Route path="/validation/satellite/typeofdamage" element={<QuizPage task={"satellite"} phase={"val"}/>} />
        <Route path="/validation/drone/typeofdamage" element={<QuizPage task={"drone-damage"} phase={"val"}/>} />
        <Route path="/loading" element={<DummyQuizPage/>}/>
        <Route path="/results" element={<ResultPage/>}/>
      </Routes>
    </AuthProvider>
  );
};

export default App;
