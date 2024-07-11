import React from 'react';
import { useNavigate } from 'react-router-dom';
import { doc, updateDoc } from 'firebase/firestore';
import { db } from '../services/firebaseConfig';
import { useAuth } from '../context/AuthContext';
import '../styles/HomePage.css';

const HomePage = () => {
  const navigate = useNavigate();
  const { user } = useAuth();

  const handleAgree = async () => {
    if (user) {
      const userDoc = doc(db, 'users', user.uid);
      await updateDoc(userDoc, {
        consent_agree: true
      });
      navigate('/training/victim/checkaid', { state: { task: 'info', phase: 'train' } });
    }
  };

  return (
    <div className="home-page">
      <div className="message-box">
        <h1>Welcome to the Quiz Game</h1>
        <p>We appreciate your participation in this survey.</p>
        <p>By clicking "Agree", you consent to be a part of this survey.</p>
        <button onClick={handleAgree} className="agree-button">Agree</button>
      </div>
    </div>
  );
};

export default HomePage;
