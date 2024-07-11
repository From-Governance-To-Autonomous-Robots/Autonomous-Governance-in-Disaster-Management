import { useState, useEffect } from 'react';
import { collection, query, where, getDocs } from 'firebase/firestore';
import { db } from '../services/firebaseConfig';

const useFirestore = (task,phase) => {
  const [question, setQuestion] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchQuestions = async () => {
      setLoading(true);
      const q = query(collection(db, 'questions'), where('task', '==', task), where('phase', '==', phase));
      const querySnapshot = await getDocs(q);
      const questionsList = [];
      querySnapshot.forEach((doc) => {
        questionsList.push({ id: doc.id, ...doc.data() });
      });
      
      if (questionsList.length > 0) {
        const randomIndex = Math.floor(Math.random() * questionsList.length);
        setQuestion(questionsList[randomIndex]);
      }
      
      setLoading(false);
    };

    fetchQuestions();
  }, [task,phase]);

  const storedQuestion = question;
  return { storedQuestion, loading };
};

export default useFirestore;
