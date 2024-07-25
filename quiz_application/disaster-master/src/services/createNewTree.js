export const createNewTree = (treeLevel) => {
    return {
      [`tree_${treeLevel}`]: {
        points: [],
        tree_score: 0,
        tree_done: false,
        isCorrectlyAnswered: 0,
        isWronglyAnswered: 0,
        isGatherAdditionalDataRequested: 0,
        isCompleted: [false, false, false, false, false],
        gotWronglyAnswered: [false, false, false, false, false],
        gotGatherAdditionalDataRequested: [false, false, false, false, false],
        tree: {
          info: {
            user_answer: [],
            availableAdditionalData: 5,
            question_id: [],
          },
          human: {
            user_answer: [],
            availableAdditionalData: 5,
            question_id: [],
          },
          damage: {
            user_answer: [],
            availableAdditionalData: 5,
            question_id: [],
          },
          satellite: {
            user_answer: [],
            availableAdditionalData: 5,
            question_id: [],
          },
          'drone-damage': {
            user_answer: [],
            availableAdditionalData: 5,
            question_id: [],
          }
        }
      }
    };
  }
  