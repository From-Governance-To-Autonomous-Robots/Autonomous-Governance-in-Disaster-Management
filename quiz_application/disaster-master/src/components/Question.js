import React from 'react';

const Question = ({ question, handleAnswer }) => {
  return (
    <div>
      <img src={question.image} alt="question" />
      <p>{question.text}</p>
      {question.question_options.map((option, index) => (
        <button key={index} onClick={() => handleAnswer(question.id, option)}>
          {option}
        </button>
      ))}
    </div>
  );
};

export default Question;
