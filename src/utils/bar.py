"""
src/utils/bar.py
================
Barra de progreso en consola con ETA en tiempo real.

No hay valores hardcodeados. El tiempo se mide en cada paso y el ETA se
recalcula dinámicamente después de cada `next()`.

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
        Nombre de la tarea (queda al final de la línea al terminar).
    bar_width : int
        Cantidad de caracteres de la barra.
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
        self._current_step   = 0          # pasos completados (0..total)
        self._step_labels    = []         # etiqueta de cada paso completado
        self._step_durations = []         # duración en segundos de cada paso
        self._last_t0        = None       # marca de tiempo del paso en curso

    # ── Ciclo de vida ──────────────────────────────────────────────────────────

    def start(self):
        """Inicia la barra. Llamar una sola vez antes del primer `next()`."""
        self._started        = True
        self._finished       = False
        self._current_step   = 0
        self._step_labels    = []
        self._step_durations = []
        self._last_t0        = time.time()
        self._render()

    def next(self, label: str = "", duration: float | None = None):
        """
        Avanza un paso.

        Parámetros
        ----------
        label : str
            Etiqueta del paso que acaba de terminar.
        duration : float | None
            Duración del paso en segundos.  Si es None se mide internamente
            desde la llamada anterior a `next()` o desde `start()`.
        """
        if not self._started or self._finished:
            return

        # Duración del paso recién finalizado
        if duration is None:
            duration = time.time() - (self._last_t0 or time.time())
        self._step_durations.append(duration)
        self._step_labels.append(label)

        self._current_step += 1
        self._last_t0       = time.time()

        self._render()

    def finish(self):
        """Finaliza la barra, limpia la línea completa."""
        self._finished = True
        # Escribir espacios para limpiar caracteres residuales
        sys.stdout.write("\r" + " " * 140 + "\r")
        sys.stdout.flush()
        # Mostrar línea final
        print(f"  █{'█' * self.bar_width}  {self.label} — completado")

    # ── Cálculos ───────────────────────────────────────────────────────────────

    def _eta(self) -> tuple[float, str]:
        """
        Calcula (segundos_restantes, texto_eta).

        Usa el promedio de las últimas 5 duraciones medidas.  Apenas hay
        suficientes datos, el estimado se va afinando automáticamente.
        """
        if not self._step_durations:
            return 0.0, "—"

        recent = self._step_durations[-5:]     # últimos 5 pasos
        avg    = sum(recent) / len(recent)
        rest   = self.total - self._current_step
        secs   = max(0.0, avg * rest)
        return secs, self._fmt_time(secs)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Formatea segundos como 'X min Y s' o 'Z s'."""
        seconds = max(0, int(round(seconds)))
        if seconds >= 60:
            m, s = divmod(seconds, 60)
            return f"{m} min {s} s"
        return f"{seconds} s"

    # ── Renderizado ────────────────────────────────────────────────────────────

    def _bar_str(self) -> str:
        """Genera la cadena visual de la barra."""
        if self.total == 0:
            return "░" * self.bar_width
        filled = int(round(self.bar_width * self._current_step / self.total))
        filled = min(filled, self.bar_width)
        return "█" * filled + "░" * (self.bar_width - filled)

    def _render(self):
        """Dibuja una línea de la barra en la consola (usa \r para overwrite)."""
        bar             = self._bar_str()
        eta_secs, eta   = self._eta()
        last_label      = self._step_labels[-1] if self._step_labels else ""

        # Línea durante la ejecución
        line = (
            f"\r  [{bar}]  {self._current_step}/{self.total}  "
            f"{last_label:<28}  ETA: {eta}"
        )

        sys.stdout.write(line)
        sys.stdout.flush()
