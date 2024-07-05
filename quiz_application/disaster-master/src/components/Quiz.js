import React, { useState } from 'react';
import useFirestore from '../hooks/useFirestore';
import Question from './Question';
import Loader from './Loader';
import Score from './Score';

const Quiz = ({ task }) => {
  const { questions, loading } = useFirestore(task);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState({});
  const [score, setScore] = useState(0);

  const handleAnswer = (questionId, answer) => {
    const correctAnswer = questions[currentQuestionIndex].correct_answer;
    setUserAnswers({ ...userAnswers, [questionId]: answer });
    if (correctAnswer === answer) {
      setScore(score + 1);
    }
    setCurrentQuestionIndex(currentQuestionIndex + 1);
  };

  if (loading) {
    return <Loader />;
  }

  if (currentQuestionIndex >= questions.length) {
    return <Score score={score} total={questions.length} />;
  }

  const currentQuestion = questions[currentQuestionIndex];

  return (
    <div>
      <Question
        question={currentQuestion}
        handleAnswer={handleAnswer}
      />
    </div>
  );
};

export default Quiz;
