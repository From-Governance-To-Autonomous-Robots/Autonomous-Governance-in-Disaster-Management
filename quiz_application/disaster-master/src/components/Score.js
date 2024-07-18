import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { doc, getDoc, updateDoc } from 'firebase/firestore';
import { db } from '../services/firebaseConfig';

const Score = () => {
  const [score, setScore] = useState(null);
  const { user } = useAuth();

  const fetchScoreData = async () => {
    const userDoc = doc(db, 'users', user.uid);
    const userDocSnapshot = await getDoc(userDoc);
    const userDocData = userDocSnapshot.data();
    // let tree0Score = userDocData.responses['tree_0'].points // this is an array , sum it up 
    // let tree1Score = userDocData.responses['tree_1'].points // this is an array , sum it up 
    // let tree2Score = userDocData.responses['tree_2'].points // this is an array , sum it up 
    // let tree3Score = userDocData.responses['tree_3'].points // this is an array , sum it up 
    // let tree4Score = userDocData.responses['tree_4'].points // this is an array , sum it up 
    // let aggregateScore = tree0Score + tree1Score + tree2Score + tree3Score + tree4Score;

    // await updateDoc(userDoc,{
    //   score: aggregateScore
    // });

    setScore(userDocData.score * 100);
  };

  useEffect(() => {
    fetchScoreData();
  }, [user.uid]);
  
  return (
    <div>
      <h2>You scored : {score} / 100</h2>
    </div>
  );
};

export default Score;
