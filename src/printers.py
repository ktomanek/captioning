# different caption printers
import os
import shutil
import sys


class CaptionPrinter:
    """Base class for caption printers."""

    def start(self):
        pass

    def stop(self):
        pass

    def print(self, transcript, duration=None, partial=False):
        raise NotImplementedError("This method should be overridden by subclasses.")

class PlainCaptionPrinter(CaptionPrinter):
    def start(self):
        print("---------------------------  Transcribing speech ----------------------------")

    def stop(self):
        print("-----------------------------------------------------------------------------")

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
            "segment": "bold blue",
        })
        self.console = Console(theme=caption_theme, highlight=False)

        self.console.rule("[bold magenta]Initializing ...")

    def start(self):
        os.system('clear')
        self.console.rule("[bold magenta]Transcribing speech...")

    def stop(self):
        self.console.rule()

    def _map_probabilities(self, p):
        """Map probabilities to colors"""
        if p > 0.9:
            return 'green'
        elif p > 0.7:
            return 'yellow'
        else:
            return 'red'


    def _maybe_colorize_transcript_with_probabilities(self, transcript, add_probabilities=False):
        """If transript contains probabilities, format them with colors.
        
        Probabilities are expected to be in the format "word/p" where p is a float.
        """
        # format probabilites
        words = transcript.split()
        for i, word in enumerate(words):
            if '/' in word:
                parts = word.split('/')
                w = parts[0].strip()
                p = float(parts[1].strip())
                color = self._map_probabilities(p)
                if add_probabilities:
                    words[i] = f"[{color}]{parts[0]}[/{color}]/[bold magenta]{parts[1]}[/bold magenta]"
                else:
                    words[i] = f"[{color}]{parts[0]}[/{color}]"
        transcript = ' '.join(words)

        return transcript

    def print(self, transcript, duration=None, partial=False):
        """Update the caption display with the latest transcription"""
        # Move to the beginning of the line and clear it
        sys.stdout.write("\r\033[K")  


        # color code probabilities if contained in transcript
        if '/' in transcript:
            transcript = self._maybe_colorize_transcript_with_probabilities(transcript, add_probabilities=False)

        if duration:
            text = f"{transcript} ({duration:.2f} sec)"
        else:
            text = transcript

        # Show partial and full segments differently
        if partial:
            # if text longer than terminal width, truncate it on the left
            terminal_width = self.console.width
            if len(text) > terminal_width/2:
                last_chars = terminal_width - 5
                text = '...' + text[-last_chars:]
            syle = "partial"
            self.console.print(text, end="", style=syle)   # Print the styled text without adding a new line
        else:
            syle = "segment"
            self.console.print(text, style=syle) # Print the styled text without adding a new line
