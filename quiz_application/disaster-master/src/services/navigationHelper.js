export const checkNavigationHelper = (task, phase) => {
    if (phase === "train") {
      if (task === "info") {
        return { path: "/training/victim/typeofaid", state: { task: 'human', phase: 'train' } };
      } else if (task === "human") {
        return { path: "/training/victim/typeofdamage", state: { task: 'damage', phase: 'train' } };
      } else if (task === "damage") {
        return { path: "/training/satellite/typeofdamage", state: { task: 'satellite', phase: 'train' } };
      } else if (task === "satellite") {
        return { path: "/training/drone/typeofdamage", state: { task: 'drone-damage', phase: 'train' } };
      } else if (task === "drone-damage") {
        return { path: "/training/victim/checkaid", state: { task: 'info', phase: 'train' } };
      } else {
        return { path: "/training/victim/checkaid", state: { task: 'info', phase: 'train' } };
      }
    } else if (phase === "val") {
      if (task === "info") {
        return { path: "/validation/victim/typeofaid", state: { task: 'human', phase: 'val' } };
      } else if (task === "human") {
        return { path: "/validation/victim/typeofdamage", state: { task: 'damage', phase: 'val' } };
      } else if (task === "damage") {
        return { path: "/validation/satellite/typeofdamage", state: { task: 'satellite', phase: 'val' } };
      } else if (task === "satellite") {
        return { path: "/validation/drone/typeofdamage", state: { task: 'drone-damage', phase: 'val' } };
      } else if (task === "drone-damage") {
        return { path: "/validation/victim/checkaid", state: { task: 'info', phase: 'val' } };
      } else {
        return { path: "/validation/victim/checkaid", state: { task: 'info', phase: 'val' } };
      }
    }
  };
  