class Gym:
    def step(self, action) -> (object, float, bool):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError
