# different caption printers

import sys

class CaptionPrinter:
    """Base class for caption printers."""
    def print(self, transcript, duration=None, partial=False):
        raise NotImplementedError("This method should be overridden by subclasses.")

class PlainCaptionPrinter(CaptionPrinter):
    def print(self, transcript, duration=None, partial=False):
        """Update the caption display with the latest transcription"""
        if partial:
            print(f"\rPARTIAL: {transcript}", flush=True, end='')
        else:
            if duration:
                print(f"\rSEGMENT, {duration:.2f} sec: {transcript}")
            else:
                print(f"\rSEGMENT: {transcript}")

class RichCaptionPrinter(CaptionPrinter):

    def __init__(self):
        # https://rich.readthedocs.io/en/stable/style.html
        from rich.console import Console
        from rich.theme import Theme
        caption_theme = Theme({
            "partial": "italic",
            "segment": "bold reverse",
        })
        self.console = Console(theme=caption_theme)

    def print(self, transcript, duration=None, partial=False):
        """Update the caption display with the latest transcription"""
        # Move to the beginning of the line and clear it
        sys.stdout.write("\r\033[K")  

        if duration:
            text = f"{transcript} ({duration:.2f} sec)"
        else:
            text = transcript

        # Show partial and full segments differently
        if partial:
            syle = "partial"
            self.console.print(text, end="", style=syle)   # Print the styled text without adding a new line
        else:
            syle = "segment" #"bold"
            self.console.print(text, style=syle)   # Print the styled text without adding a new line
