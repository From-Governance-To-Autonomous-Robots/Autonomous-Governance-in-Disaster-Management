import { useState, useEffect } from 'react';
import { collection, query, where, getDocs } from 'firebase/firestore';
import { db } from '../services/firebaseConfig';

const useHelperData = (task, phase) => {
  const [helperData, setHelperData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHelperData = async () => {
      setLoading(true);
      const q = query(collection(db, 'questions'), where('task', '==', task), where('phase', '==', phase));
      const querySnapshot = await getDocs(q);
      const helperDataList = {};

      querySnapshot.forEach((doc) => {
        const data = doc.data();
        const correctAnswer = data.correct_answer;

        if (!helperDataList[correctAnswer]) {
          helperDataList[correctAnswer] = [];
        }
        helperDataList[correctAnswer].push({ id: doc.id, ...data });
      });

      const randomHelperData = Object.keys(helperDataList).map((key) => {
        const dataList = helperDataList[key];
        const randomIndex = Math.floor(Math.random() * dataList.length);
        return dataList[randomIndex];
      });

      setHelperData(randomHelperData);
      setLoading(false);
    };

    fetchHelperData();
  }, [task, phase]);

  return { helperData, loading };
};

export default useHelperData;