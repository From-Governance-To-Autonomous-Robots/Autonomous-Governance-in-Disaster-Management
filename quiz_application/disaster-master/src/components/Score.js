import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { doc, getDoc } from 'firebase/firestore';
import { db } from '../services/firebaseConfig';
import '../styles/Score.css'; 

const Score = () => {
  const [score, setScore] = useState(null);
  const { user } = useAuth();

  const fetchScoreData = async () => {
    const userDoc = doc(db, 'users', user.uid);
    const userDocSnapshot = await getDoc(userDoc);
    const userDocData = userDocSnapshot.data();

    setScore(userDocData.score * 100);
  };

  useEffect(() => {
    fetchScoreData();
  }, [user.uid]);

  return (
    <div className="score-container1">
      <div className="score-display1">
        <h2>You Scored:</h2>
        <h1 className="score-value1">{score} / 100</h1>
      </div>
    </div>
  );
};

export default Score;
