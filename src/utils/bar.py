"""
src/utils/bar.py
================
Barra de progreso en consola con ETA en tiempo real.

No hay valores hardcodeados. El tiempo se mide en cada paso y el ETA se
recalcula dinámicamente después de cada `next()`.

Formato en pantalla:
  [■■■■░░░░] 4/9 | s04_reacciones.py            | Transcurrido: 32 s | Restante: ~45 s

Uso:
    bar = ProgressBar(total=9, label="EDA")
    bar.start()
    for paso in range(9):
        ejecutar_paso_actual()
        bar.next(label=f"Paso {paso+1}")
    bar.finish()
"""

import sys
import time


class ProgressBar:
    """
    Barra de progreso de consola sin suposiciones iniciales.

    Parámetros
    ----------
    total : int
        Cantidad total de pasos.
    label : str
        Nombre de la tarea (aparece en la línea final al terminar).
    bar_width : int
        Cantidad de caracteres de la barra (relleno + vacío).
    """

    def __init__(self, total: int, label: str = "", bar_width: int = 30):
        if total < 1:
            raise ValueError("total debe ser >= 1")
        self.total        = total
        self.label        = label
        self.bar_width    = bar_width
        self._started     = False
        self._finished    = False

        # Estado interno
        self._current_step   = 0      # pasos completados (0..total)
        self._step_labels    = []     # etiqueta de cada paso completado
        self._step_durations = []     # duración en segundos de cada paso
        self._last_t0        = None   # marca de tiempo del paso en curso
        self._global_t0      = None   # marca de tiempo del inicio total

    # ── Ciclo de vida ──────────────────────────────────────────────────────────

    def start(self):
        """Inicia la barra. Llamar una sola vez antes del primer `next()`."""
        self._started        = True
        self._finished       = False
        self._current_step   = 0
        self._step_labels    = []
        self._step_durations = []
        self._last_t0        = time.time()
        self._global_t0      = time.time()
        self._render()

    def next(self, label: str = "", duration: float | None = None):
        """
        Avanza un paso.

        Parámetros
        ----------
        label : str
            Etiqueta del paso que acaba de terminar.
        duration : float | None
            Duración del paso en segundos. Si es None se mide internamente
            desde la llamada anterior a `next()` o desde `start()`.
        """
        if not self._started or self._finished:
            return

        if duration is None:
            duration = time.time() - (self._last_t0 or time.time())
        self._step_durations.append(duration)
        self._step_labels.append(label)

        self._current_step += 1
        self._last_t0       = time.time()

        self._render()

    def finish(self):
        """Finaliza la barra y muestra la línea de resumen."""
        self._finished = True
        # Limpiar la línea de progreso
        sys.stdout.write("\r" + " " * 160 + "\r")
        sys.stdout.flush()
        # Línea final con barra llena y tiempo total
        elapsed     = time.time() - self._global_t0
        elapsed_str = self._fmt_time(elapsed)
        bar         = "■" * self.bar_width
        print(
            f"  [{bar}] {self.total}/{self.total}"
            f" | {self.label:<28}"
            f" | Transcurrido: {elapsed_str}"
            f" | Restante: ~0 s"
        )

    # ── Cálculos ───────────────────────────────────────────────────────────────

    def _eta(self) -> tuple[float, str]:
        """
        Calcula (segundos_restantes, texto_eta).

        Usa el promedio de las últimas 5 duraciones medidas. Apenas hay
        suficientes datos, el estimado se va afinando automáticamente.
        """
        if not self._step_durations:
            return 0.0, "—"

        recent = self._step_durations[-5:]
        avg    = sum(recent) / len(recent)
        rest   = self.total - self._current_step
        secs   = max(0.0, avg * rest)
        return secs, self._fmt_time(secs)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """
        Formatea un tiempo en segundos de forma legible.

        Reglas:
          - < 1 s   → muestra decimales: '0.3 s'
          - < 60 s  → entero: '8 s'
          - < 3600 s → minutos y segundos: '1 min 5 s'
          - < 86400 s → horas y minutos: '2 h 15 min'
          - >= 86400 s → días y horas: '1 d 4 h'
        """
        if seconds < 0:
            seconds = 0.0

        if seconds < 1.0:
            # Sub-segundo: un decimal
            return f"{seconds:.1f} s"

        if seconds < 60.0:
            return f"{int(round(seconds))} s"

        if seconds < 3600.0:
            m = int(seconds) // 60
            s = int(seconds) % 60
            return f"{m} min {s} s"

        if seconds < 86400.0:
            h = int(seconds) // 3600
            m = (int(seconds) % 3600) // 60
            return f"{h} h {m} min"

        d = int(seconds) // 86400
        h = (int(seconds) % 86400) // 3600
        return f"{d} d {h} h"

    # ── Renderizado ────────────────────────────────────────────────────────────

    def _bar_str(self) -> str:
        """Genera la cadena visual de la barra."""
        if self.total == 0:
            return "░" * self.bar_width
        filled = int(round(self.bar_width * self._current_step / self.total))
        filled = min(filled, self.bar_width)
        return "■" * filled + "░" * (self.bar_width - filled)

    def _render(self):
        """Dibuja la barra en la consola (sobreescribe con \\r)."""
        bar            = self._bar_str()
        _eta_secs, eta = self._eta()
        last_label     = self._step_labels[-1] if self._step_labels else ""
        elapsed        = time.time() - self._global_t0
        elapsed_str    = self._fmt_time(elapsed)
        eta_str        = f"~{eta}" if self._step_durations else "~-"

        # Formato: [■■■░░░] N/M | label              | Transcurrido: X s | Restante: ~Y s
        # Reducimos los anchos internos ligeramente para garantizar que quepa en 100 columnas
        line = (
            f"\r  [{bar}] {self._current_step}/{self.total}"
            f" | {last_label:<24}"
            f" | Transcurrido: {elapsed_str}"
            f" | Restante: {eta_str}"
        )

        # Rellenamos la línea hasta 100 caracteres. Esto limpia residuos sin provocar saltos automáticos (wrap)
        line = line.ljust(100)

        sys.stdout.write(line)
        sys.stdout.flush()
