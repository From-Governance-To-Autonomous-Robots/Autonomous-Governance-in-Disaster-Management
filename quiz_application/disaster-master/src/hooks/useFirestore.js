import { useState, useEffect } from 'react';
import { collection, query, where, getDocs } from 'firebase/firestore';
import { db } from '../services/firebaseConfig';

const useFirestore = (task) => {
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchQuestions = async () => {
      setLoading(true);
      const q = query(collection(db, 'questions'), where('task', '==', task));
      const querySnapshot = await getDocs(q);
      const questionsList = [];
      querySnapshot.forEach((doc) => {
        questionsList.push({ id: doc.id, ...doc.data() });
      });
      setQuestions(questionsList);
      setLoading(false);
    };

    fetchQuestions();
  }, [task]);

  return { questions, loading };
};

export default useFirestore;
