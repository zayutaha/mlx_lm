import unittest


class TestModelRunner(unittest.TestCase):

    def test_interrupt_not_start_uses_sigint(self):
        """interrupt() — NOT start() — should send SIGINT (not \\x04)."""
        import inspect
        from model_lifecycle import ModelRunner

        src = inspect.getsource(ModelRunner.interrupt)
        self.assertIn("SIGINT", src)
        self.assertNotIn("\\x04", src)
        self.assertNotIn("stdin.write", src)

    def test_interrupt_uses_sigint(self):
        """interrupt() should send SIGINT (not \\x04)."""
        import inspect
        from model_lifecycle import ModelRunner

        src = inspect.getsource(ModelRunner.interrupt)
        self.assertIn("SIGINT", src)
        self.assertNotIn("\\x04", src)
        self.assertNotIn("stdin.write", src)


if __name__ == "__main__":
    unittest.main()
