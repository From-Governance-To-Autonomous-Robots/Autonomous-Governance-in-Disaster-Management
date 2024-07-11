import React, { useEffect, useState } from 'react';
import Quiz from '../components/Quiz';
import { useLocation } from 'react-router-dom';

const QuizPage = () => {
  const location = useLocation();
  const { task, phase } = location.state || {};

  console.log('task : ',task)
  console.log('phase : ',phase)

  return (
    <div>
      <Quiz task={task} phase={phase} />
    </div>
  );
};

export default QuizPage;
