import React from 'react';

const Score = ({ score, total }) => {
  return (
    <div>
      <h2>Your score: {score} / {total}</h2>
    </div>
  );
};

export default Score;
