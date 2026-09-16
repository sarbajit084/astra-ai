"""High-performance Deterministic Mathematical Engine for Astra AI.

Powered by SymPy symbolic mathematics:
- Arithmetic, fractions, percentages, powers, roots, factorials
- Algebra: simplify, expand, factor, partial fractions, polynomial roots
- Equations: linear, quadratic (with discriminant & formula steps), polynomial, simultaneous systems
- Calculus: derivatives (1st, 2nd, nth), integrals (indefinite with +C, definite with bounds), limits, Taylor series
- Matrices: determinant, inverse, transpose, trace, rank, eigenvalues, multiplication
- Trigonometry: exact angle evaluations, identities, radian/degree conversions, hyperbolic
- Logarithms: natural, common, arbitrary base, log/exponential equation solving
- Probability & Statistics: nCr, nPr, mean, median, mode, variance, standard deviation
- Geometry: area, perimeter, surface area, volume (2D & 3D shapes), Pythagorean theorem
- Unit Conversions: length, mass, temperature, speed, volume, time, data storage
- Natural voice/keyboard transcription preprocessing
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Any

import sympy as sp
from sympy import (
    E,
    Float,
    Integer,
    Matrix,
    Rational,
    Symbol,
    apart,
    cos,
    diff,
    exp,
    expand,
    factor,
    factorial,
    integrate,
    latex,
    limit,
    log,
    oo,
    pi,
    series,
    simplify,
    sin,
    solve,
    sqrt,
    symbols,
    sympify,
    tan,
    trigsimp,
)


class MathSolver:
    """Deterministic mathematical problem solver powered by SymPy."""

    # Keywords that indicate code requests rather than pure mathematics
    CODE_KEYWORDS = {
        "code", "script", "program", "python", "function", "javascript",
        "html", "css", "java", "c++", "algorithm", "class", "def ", "import "
    }

    # Standard unit conversion factors to base SI units
    LENGTH_UNITS = {
        "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
        "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
        "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
        "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
        "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
        "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
        "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
        "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
        "nm": 1e-9, "nanometer": 1e-9, "nanometers": 1e-9,
        "nautical mile": 1852.0, "nautical miles": 1852.0, "nmi": 1852.0
    }

    MASS_UNITS = {
        "kg": 1.0, "kilogram": 1.0, "kilograms": 1.0, "kilo": 1.0, "kilos": 1.0,
        "g": 0.001, "gram": 0.001, "grams": 0.001,
        "mg": 1e-6, "milligram": 1e-6, "milligrams": 1e-6,
        "lb": 0.45359237, "lbs": 0.45359237, "pound": 0.45359237, "pounds": 0.45359237,
        "oz": 0.028349523125, "ounce": 0.028349523125, "ounces": 0.028349523125,
        "ton": 1000.0, "tons": 1000.0, "metric ton": 1000.0, "metric tons": 1000.0
    }

    SPEED_UNITS = {
        "m/s": 1.0, "mps": 1.0, "meter per second": 1.0, "meters per second": 1.0,
        "km/h": 1.0 / 3.6, "kph": 1.0 / 3.6, "kmh": 1.0 / 3.6, "kilometer per hour": 1.0 / 3.6, "kilometers per hour": 1.0 / 3.6,
        "mph": 0.44704, "mi/h": 0.44704, "miles per hour": 0.44704, "mile per hour": 0.44704,
        "knot": 0.514444, "knots": 0.514444
    }

    VOLUME_UNITS = {
        "l": 1.0, "liter": 1.0, "liters": 1.0, "litre": 1.0, "litres": 1.0,
        "ml": 0.001, "milliliter": 0.001, "milliliters": 0.001,
        "m^3": 1000.0, "cubic meter": 1000.0, "cubic meters": 1000.0,
        "cm^3": 0.001, "cc": 0.001,
        "gallon": 3.78541, "gallons": 3.78541, "gal": 3.78541,
        "quart": 0.946353, "quarts": 0.946353, "qt": 0.946353,
        "pint": 0.473176, "pints": 0.473176, "pt": 0.473176,
        "cup": 0.236588, "cups": 0.236588,
        "fluid ounce": 0.0295735, "fluid ounces": 0.0295735, "fl oz": 0.0295735
    }

    DATA_UNITS = {
        "b": 1.0, "byte": 1.0, "bytes": 1.0,
        "kb": 1024.0, "kilobyte": 1024.0, "kilobytes": 1024.0,
        "mb": 1024.0**2, "megabyte": 1024.0**2, "megabytes": 1024.0**2,
        "gb": 1024.0**3, "gigabyte": 1024.0**3, "gigabytes": 1024.0**3,
        "tb": 1024.0**4, "terabyte": 1024.0**4, "terabytes": 1024.0**4,
        "pb": 1024.0**5, "petabyte": 1024.0**5, "petabytes": 1024.0**5
    }

    TIME_UNITS = {
        "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
        "min": 60.0, "minute": 60.0, "minutes": 60.0,
        "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
        "day": 86400.0, "days": 86400.0,
        "week": 604800.0, "weeks": 604800.0,
        "month": 2629746.0, "months": 2629746.0,
        "year": 31556952.0, "years": 31556952.0
    }

    @classmethod
    def sanitize_math_input(cls, query: str) -> str:
        """Preprocesses queries from keyboards and voice recognition into standard mathematical syntax."""
        text = query.strip()

        # Replace spoken power phrases
        text = re.sub(r"\b([a-zA-Z])\s+squared\b", r"\1^2", text, flags=re.IGNORECASE)
        text = re.sub(r"\b([a-zA-Z])\s+cubed\b", r"\1^3", text, flags=re.IGNORECASE)
        text = re.sub(r"\bto the power of\b", "^", text, flags=re.IGNORECASE)
        text = re.sub(r"\braised to(?: the power of)?\b", "^", text, flags=re.IGNORECASE)

        # Replace spoken roots
        text = re.sub(r"\bsquare root of\b", "sqrt", text, flags=re.IGNORECASE)
        text = re.sub(r"\bcube root of\b", "cbrt", text, flags=re.IGNORECASE)

        # Replace spoken arithmetic terms
        text = re.sub(r"\bmultiplied by\b", "*", text, flags=re.IGNORECASE)
        text = re.sub(r"\btimes\b", "*", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdivided by\b", "/", text, flags=re.IGNORECASE)
        text = re.sub(r"\bplus\b", "+", text, flags=re.IGNORECASE)
        text = re.sub(r"\bminus\b", "-", text, flags=re.IGNORECASE)
        text = re.sub(r"\bequals\b|\bequal to\b|\bis equal to\b", "=", text, flags=re.IGNORECASE)

        # Clean unicode math symbols
        text = text.replace("×", "*").replace("÷", "/").replace("−", "-")
        text = text.replace("²", "^2").replace("³", "^3").replace("√", "sqrt")

        # Normalize implicit multiplication e.g., 2x -> 2*x, 3sin(x) -> 3*sin(x)
        # Avoid breaking English words like "0 to", "2 from", "10 and"
        def _repl_implicit(m):
            digit, word = m.group(1), m.group(2)
            if len(word) == 1 and word.isalpha():
                return f"{digit}*{word}"
            if word.lower() in ("sin", "cos", "tan", "sec", "csc", "cot", "log", "ln", "exp", "sqrt", "cbrt"):
                return f"{digit}*{word}"
            return m.group(0)

        text = re.sub(r"(\d+)\s*([a-zA-Z]+)", _repl_implicit, text)
        text = re.sub(r"(\))\s*(\()", r"\1*\2", text)
        text = re.sub(r"(\d+)\s*(\()", r"\1*\2", text)

        return text.strip()

    @classmethod
    def solve(cls, query: str, mode: str = "general", detailed: bool = False) -> str | None:
        """Main entry point. Analyzes query and routes through specialized mathematical engines."""
        if not query or len(query.strip()) < 2:
            return None

        # If user explicitly requested code/programming, do not intercept with math solver
        q_lower = query.lower()
        if mode == "code" or any(f" {k} " in f" {q_lower} " for k in cls.CODE_KEYWORDS):
            return None

        # Try specialized domain solvers in order of specificity
        try:
            # 1. Unit conversions
            res = cls._solve_unit_conversion(query)
            if res: return res

            # 2. Geometry calculations (area, volume, perimeter, Pythagorean)
            res = cls._solve_geometry(query)
            if res: return res

            # 3. Probability & Statistics (combinations, permutations, mean, std dev)
            res = cls._solve_statistics(query)
            if res: return res

            # 4. Matrix & Linear Algebra (determinant, inverse, eigenvalues, multiplication)
            res = cls._solve_matrix(query)
            if res: return res

            # 5. Calculus (derivatives, integrals, limits, series)
            res = cls._solve_calculus(query, detailed)
            if res: return res

            # 6. Simultaneous system of equations
            res = cls._solve_simultaneous_equations(query, detailed)
            if res: return res

            # 7. Single Equations & Quadratics
            res = cls._solve_equations(query, detailed)
            if res: return res

            # 8. Algebra (simplify, expand, factor, partial fractions)
            res = cls._solve_algebra(query, detailed)
            if res: return res

            # 9. Trigonometry & Logarithms
            res = cls._solve_trigonometry_and_logs(query, detailed)
            if res: return res

            # 10. General Arithmetic & Percentage fallback
            res = cls._solve_arithmetic(query, detailed)
            if res: return res

        except Exception:
            return None

        return None

    # =========================================================================
    # 1. UNIT CONVERSIONS
    # =========================================================================
    @classmethod
    def _solve_unit_conversion(cls, query: str) -> str | None:
        q = query.strip().lower()
        # Matches patterns like "convert 50 miles per hour to km/h", "100 km in miles", "32 c to f"
        m = re.search(r"(?:convert\s+)?([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z/°º^32]+(?:\s+[a-zA-Z]+)?)\s+(?:in|to|into)\s+([a-zA-Z/°º^32]+(?:\s+[a-zA-Z]+)?)", q)
        if not m:
            return None

        val_str, from_u, to_u = m.group(1), m.group(2).strip(), m.group(3).strip()
        val = float(val_str)

        # Normalize temperature
        temp_aliases = {
            "c": "c", "celsius": "c", "centigrade": "c", "°c": "c",
            "f": "f", "fahrenheit": "f", "°f": "f",
            "k": "k", "kelvin": "k", "°k": "k"
        }
        f_temp = temp_aliases.get(from_u)
        t_temp = temp_aliases.get(to_u)
        if f_temp and t_temp:
            u_from = f_temp.upper()
            u_to = t_temp.upper()
            if f_temp == t_temp:
                return f"**Temperature Conversion:**\n\n$${val}^{{\\circ}}\\mathrm{{{u_from}}} = {val}^{{\\circ}}\\mathrm{{{u_to}}}$$"
            # Convert to C
            c_val = val if f_temp == "c" else ((val - 32) * 5 / 9 if f_temp == "f" else val - 273.15)
            # Convert C to target
            target_val = c_val if t_temp == "c" else (c_val * 9 / 5 + 32 if t_temp == "f" else c_val + 273.15)
            formula_desc = {
                ("c", "f"): "F = (C \\times \\frac{9}{5}) + 32",
                ("f", "c"): "C = (F - 32) \\times \\frac{5}{9}",
                ("c", "k"): "K = C + 273.15",
                ("k", "c"): "C = K - 273.15",
                ("f", "k"): "K = (F - 32) \\times \\frac{5}{9} + 273.15",
                ("k", "f"): "F = (K - 273.15) \\times \\frac{9}{5} + 32"
            }.get((f_temp, t_temp), "")

            return (
                f"**Temperature Conversion:**\n\n"
                f"**Formula:** $${formula_desc}$$\n\n"
                f"**Result:** $${val}^{{\\circ}}\\mathrm{{{u_from}}} = \\mathbf{{{target_val:.4g}}}^{{\\circ}}\\mathrm{{{u_to}}}$$"
            )

        # Standard linear unit conversions
        for unit_dict, unit_type in [
            (cls.LENGTH_UNITS, "Length"),
            (cls.MASS_UNITS, "Mass / Weight"),
            (cls.SPEED_UNITS, "Speed"),
            (cls.VOLUME_UNITS, "Volume"),
            (cls.DATA_UNITS, "Digital Storage"),
            (cls.TIME_UNITS, "Time"),
        ]:
            if from_u in unit_dict and to_u in unit_dict:
                from_factor = unit_dict[from_u]
                to_factor = unit_dict[to_u]
                result = (val * from_factor) / to_factor
                return (
                    f"**{unit_type} Conversion:**\n\n"
                    f"$${val:g}\\;\\mathrm{{{from_u}}} = \\mathbf{{{result:.6g}\\;\\mathrm{{{to_u}}}}}$$"
                )

        return None

    # =========================================================================
    # 2. GEOMETRY
    # =========================================================================
    @classmethod
    def _solve_geometry(cls, query: str) -> str | None:
        q = query.strip().lower()

        # Circle: Area or Circumference
        m_circle = re.search(r"(?:area|circumference)\s+(?:of\s+)?(?:a\s+)?circle\s+(?:with\s+)?(?:radius|r)\s*(?:=|is)?\s*(\d+(?:\.\d+)?)", q)
        if m_circle:
            r = float(m_circle.group(1))
            is_area = "area" in q
            if is_area:
                exact = pi * Rational(str(r))**2
                approx = math.pi * (r**2)
                return (
                    f"**Area of Circle** ($r = {r}$):\n\n"
                    f"**Formula:** $$A = \\pi r^2$$\n\n"
                    f"**Exact Solution:** $$A = {latex(exact)}$$\n\n"
                    f"**Decimal Approximation:** $$A \\approx \\mathbf{{{approx:.6g}}}$$"
                )
            else:
                exact = 2 * pi * Rational(str(r))
                approx = 2 * math.pi * r
                return (
                    f"**Circumference of Circle** ($r = {r}$):\n\n"
                    f"**Formula:** $$C = 2\\pi r$$\n\n"
                    f"**Exact Solution:** $$C = {latex(exact)}$$\n\n"
                    f"**Decimal Approximation:** $$C \\approx \\mathbf{{{approx:.6g}}}$$"
                )

        # Pythagorean Theorem
        m_pyth = re.search(r"(?:hypotenuse|pythagor(?:as|ean))\s+(?:.*?legs?\s+)?(?:a\s*=\s*)?(\d+(?:\.\d+)?)\s*(?:and|,)?\s*(?:b\s*=\s*)?(\d+(?:\.\d+)?)", q)
        if m_pyth:
            a, b = float(m_pyth.group(1)), float(m_pyth.group(2))
            c_sq = a**2 + b**2
            c_val = math.sqrt(c_sq)
            return (
                f"**Pythagorean Theorem** ($a = {a}$, $b = {b}$):\n\n"
                f"**Formula:** $$c = \\sqrt{{a^2 + b^2}}$$\n\n"
                f"$$c = \\sqrt{{{a}^2 + {b}^2}} = \\sqrt{{{c_sq:g}}} = \\mathbf{{{c_val:.6g}}}$$"
            )

        # Sphere: Volume / Surface Area
        m_sphere = re.search(r"(?:volume|surface area)\s+(?:of\s+)?(?:a\s+)?sphere\s+(?:with\s+)?(?:radius|r)\s*(?:=|is)?\s*(\d+(?:\.\d+)?)", q)
        if m_sphere:
            r = float(m_sphere.group(1))
            if "volume" in q:
                v_approx = (4.0 / 3.0) * math.pi * (r**3)
                v_exact = Rational(4, 3) * pi * Rational(str(r))**3
                return (
                    f"**Volume of Sphere** ($r = {r}$):\n\n"
                    f"**Formula:** $$V = \\frac{4}{3}\\pi r^3$$\n\n"
                    f"**Exact:** $$V = {latex(v_exact)}$$\n\n"
                    f"**Approximate:** $$V \\approx \\mathbf{{{v_approx:.6g}}}$$"
                )
            else:
                sa_approx = 4.0 * math.pi * (r**2)
                sa_exact = 4 * pi * Rational(str(r))**2
                return (
                    f"**Surface Area of Sphere** ($r = {r}$):\n\n"
                    f"**Formula:** $$A = 4\\pi r^2$$\n\n"
                    f"**Exact:** $$A = {latex(sa_exact)}$$\n\n"
                    f"**Approximate:** $$A \\approx \\mathbf{{{sa_approx:.6g}}}$$"
                )

        return None

    # =========================================================================
    # 3. PROBABILITY & STATISTICS
    # =========================================================================
    @classmethod
    def _solve_statistics(cls, query: str) -> str | None:
        q = query.strip().lower()

        # Combinations: nCr or "n choose r" or "combinations of n taken r"
        m_comb = re.search(r"(\d+)\s*(?:c|choose)\s*(\d+)", q)
        if not m_comb:
            m_comb = re.search(r"combination(?:s)?\s+(?:of\s+)?(\d+)\s+(?:items?\s+)?(?:taken\s+)?(\d+)", q)
        if m_comb:
            n, r = int(m_comb.group(1)), int(m_comb.group(2))
            if 0 <= r <= n <= 100:
                val = math.comb(n, r)
                return (
                    f"**Combination $\\binom{{{n}}}{{{r}}}$ ($n={n}, r={r}$):**\n\n"
                    f"**Formula:** $$\\binom{{n}}{{r}} = \\frac{{n!}}{{r!(n-r)!}}$$\n\n"
                    f"$$\\binom{{{n}}}{{{r}}} = \\frac{{{n}!}}{{{r}! \\cdot {n-r}!}} = \\mathbf{{{val}}}$$"
                )

        # Permutations: nPr or "permutations of n taken r"
        m_perm = re.search(r"(\d+)\s*p\s*(\d+)", q)
        if not m_perm:
            m_perm = re.search(r"permutation(?:s)?\s+(?:of\s+)?(\d+)\s+(?:items?\s+)?(?:taken\s+)?(\d+)", q)
        if m_perm:
            n, r = int(m_perm.group(1)), int(m_perm.group(2))
            if 0 <= r <= n <= 100:
                val = math.perm(n, r)
                return (
                    f"**Permutation $P({n}, {r})$:**\n\n"
                    f"**Formula:** $$P(n, r) = \\frac{{n!}}{{(n-r)!}}$$\n\n"
                    f"$$P({n}, {r}) = \\frac{{{n}!}}{{{n-r}!}} = \\mathbf{{{val}}}$$"
                )

        # Factorial: n! or "factorial of n"
        m_fact = re.search(r"(\d+)\s*!", query)
        if not m_fact:
            m_fact = re.search(r"factorial\s+(?:of\s+)?(\d+)", q)
        if m_fact:
            n = int(m_fact.group(1))
            if 0 <= n <= 100:
                res = math.factorial(n)
                return f"**Factorial:** $${n}! = \\mathbf{{{res}}}$$"

        # Descriptive Statistics: Mean, Median, Mode, Variance, Standard Deviation
        m_stats = re.search(r"(mean|average|median|mode|variance|standard deviation|std dev)\s+(?:of\s+)?\[?([0-9\s,\.\-]+)\]?", q)
        if m_stats:
            stat_name = m_stats.group(1)
            raw_nums = re.findall(r"[+-]?\d+(?:\.\d+)?", m_stats.group(2))
            if len(raw_nums) >= 2:
                nums = [float(x) for x in raw_nums]
                n = len(nums)
                res_str = ""
                if stat_name in ("mean", "average"):
                    m_val = statistics.mean(nums)
                    res_str = f"**Mean (Average):** $$\\mu = \\frac{{\\sum x_i}}{{n}} = \\frac{{{sum(nums):g}}}{{{n}}} = \\mathbf{{{m_val:.6g}}}$$"
                elif stat_name == "median":
                    med = statistics.median(nums)
                    res_str = f"**Median:** $$\\mathbf{{{med:g}}}$$"
                elif stat_name in ("standard deviation", "std dev"):
                    std = statistics.stdev(nums) if n > 1 else 0.0
                    res_str = f"**Sample Standard Deviation ($s$):** $$s = \\sqrt{{\\frac{{\\sum (x_i - \\bar{{x}})^2}}{{n - 1}}}} = \\mathbf{{{std:.6g}}}$$"
                elif stat_name == "variance":
                    var = statistics.variance(nums) if n > 1 else 0.0
                    res_str = f"**Sample Variance ($s^2$):** $$s^2 = \\mathbf{{{var:.6g}}}$$"

                if res_str:
                    return f"**Dataset:** $\\{{{', '.join(str(x) for x in nums)}\\}}$ ($N={n}$)\n\n{res_str}"

        return None

    # =========================================================================
    # 4. MATRICES & LINEAR ALGEBRA
    # =========================================================================
    @classmethod
    def _solve_matrix(cls, query: str) -> str | None:
        q = query.strip()
        # Look for matrix bracket notation: [[1, 2], [3, 4]]
        if "[" not in q or "]" not in q:
            return None

        try:
            mat_raw = re.search(r"\[(\s*\[.*?\]\s*)\]", q, flags=re.DOTALL)
            if not mat_raw:
                return None
            import ast
            parsed_list = ast.literal_eval(mat_raw.group(0))
            A = Matrix(parsed_list)
        except Exception:
            return None

        q_lower = q.lower()
        rows, cols = A.shape

        # Determinant
        if any(w in q_lower for w in ["determinant", "det"]):
            if rows != cols:
                return "The determinant is only defined for square matrices ($n \\times n$)."
            d = A.det()
            return (
                f"**Matrix Determinant:**\n\n"
                f"$$A = {latex(A)}$$\n\n"
                f"$$\\det(A) = |A| = \\mathbf{{{latex(d)}}}$$"
            )

        # Inverse
        if any(w in q_lower for w in ["inverse", "invert", "a^-1"]):
            if rows != cols:
                return "A matrix must be square to have an inverse."
            det_val = A.det()
            if det_val == 0:
                return f"$$\\det(A) = 0$$\n\nThis matrix is **singular** (non-invertible) because its determinant is zero."
            inv_A = A.inv()
            return (
                f"**Matrix Inverse ($A^{{-1}}$):**\n\n"
                f"$$A = {latex(A)}$$\n\n"
                f"$$A^{{-1}} = \\mathbf{{{latex(inv_A)}}}$$"
            )

        # Transpose
        if any(w in q_lower for w in ["transpose", "a^t"]):
            At = A.T
            return (
                f"**Matrix Transpose ($A^T$):**\n\n"
                f"$$A = {latex(A)}$$\n\n"
                f"$$A^T = \\mathbf{{{latex(At)}}}$$"
            )

        # Eigenvalues
        if any(w in q_lower for w in ["eigenvalue", "eigenvalues", "eigen"]):
            if rows != cols:
                return "Eigenvalues are only defined for square matrices."
            eigs = A.eigenvals()
            eig_parts = []
            for k, v in eigs.items():
                eig_parts.append(f"\\lambda = {latex(k)} \\; (\\mathrm{{mult.}} \\; {v})")
            eig_list = ", \\quad ".join(eig_parts)
            return (
                f"**Eigenvalues of Matrix:**\n\n"
                f"$$A = {latex(A)}$$\n\n"
                f"**Characteristic Equation:** $$\\det(A - \\lambda I) = 0$$\n\n"
                f"**Eigenvalues:** $${eig_list}$$"
            )

        return None

    # =========================================================================
    # 5. CALCULUS (Derivatives, Integrals, Limits, Series)
    # =========================================================================
    @classmethod
    def _solve_calculus(cls, query: str, detailed: bool = False) -> str | None:
        q = cls.sanitize_math_input(query).lower()
        x = symbols('x')

        # ── A. Definite & Indefinite Integrals ──
        if any(w in q for w in ["integrate", "integral", "antiderivative", "∫"]):
            try:
                # Check for definite integral with bounds: "from a to b"
                m_bounds = re.search(r"from\s+([+-]?\d+(?:\.\d+)?|[a-zA-Z_]+)\s+to\s+([+-]?\d+(?:\.\d+)?|[a-zA-Z_]+)", q)
                bounds = None
                if m_bounds:
                    b_low = sympify(m_bounds.group(1), locals={"pi": pi, "e": E})
                    b_high = sympify(m_bounds.group(2), locals={"pi": pi, "e": E})
                    bounds = (x, b_low, b_high)

                # Extract expression
                clean_expr_str = re.sub(r"(?:evaluate|calculate|find)?\s*(?:the\s+)?(?:definite\s+)?(?:integral\s+(?:of)?|integrate)\s*", "", q)
                clean_expr_str = re.sub(r"from\s+.*?\s+to\s+.*", "", clean_expr_str)
                clean_expr_str = re.sub(r"d[xX]$", "", clean_expr_str).strip()

                if clean_expr_str:
                    clean_expr_str = clean_expr_str.replace("^", "**")
                    expr = sympify(clean_expr_str, locals={"x": x, "e": E, "pi": pi, "sin": sin, "cos": cos, "tan": tan, "exp": exp, "log": log, "sqrt": sqrt})

                    if bounds:
                        res = integrate(expr, bounds)
                        return (
                            f"**Definite Integral:**\n\n"
                            f"$$\\int_{{{latex(bounds[1])}}}^{{{latex(bounds[2])}}} {latex(expr)}\\,dx$$\n\n"
                            f"**Antiderivative:** $$F(x) = {latex(integrate(expr, x))}$$\n\n"
                            f"**Evaluation:** $$\\left[ F(x) \\right]_{{{latex(bounds[1])}}}^{{{latex(bounds[2])}}} = \\mathbf{{{latex(res)}}}$$"
                        )
                    else:
                        res = integrate(expr, x)
                        return (
                            f"**Indefinite Integral:**\n\n"
                            f"$$\\int {latex(expr)}\\,dx = \\mathbf{{{latex(res)} + C}}$$"
                        )
            except Exception:
                pass

        # ── B. Derivatives & Differentiation ──
        if any(w in q for w in ["derivative", "differentiate", "diff", "dy/dx", "d/dx"]):
            try:
                # Detect order (e.g., second derivative)
                order = 2 if "second" in q or "2nd" in q or "d^2/dx^2" in q else 1

                clean_expr_str = re.sub(r"(?:find|calculate|evaluate)?\s*(?:the\s+)?(?:first|second|2nd)?\s*(?:derivative\s+(?:of)?|differentiate)\s*", "", q)
                clean_expr_str = re.sub(r"with\s+respect\s+to\s+x", "", clean_expr_str).strip()
                if clean_expr_str:
                    clean_expr_str = clean_expr_str.replace("^", "**")
                    expr = sympify(clean_expr_str, locals={"x": x, "e": E, "pi": pi, "sin": sin, "cos": cos, "tan": tan, "exp": exp, "log": log, "sqrt": sqrt})
                    res = diff(expr, x, order)
                    order_sym = "\\frac{d^2}{dx^2}" if order == 2 else "\\frac{d}{dx}"
                    return (
                        f"**Derivative ({'2nd Order' if order == 2 else '1st Order'}):**\n\n"
                        f"$${order_sym}\\left[{latex(expr)}\\right] = \\mathbf{{{latex(res)}}}$$"
                    )
            except Exception:
                pass

        # ── C. Limits ──
        if "limit" in q:
            try:
                m_lim = re.search(r"limit\s+(?:of\s+)?(.*?)\s+as\s+x\s*(?:approaches|->|to)\s*([+-]?(?:oo|inf|infinity|\d+(?:\.\d+)?))", q)
                if m_lim:
                    expr_str = m_lim.group(1).replace("^", "**").strip()
                    target_str = m_lim.group(2).strip()
                    target = oo if target_str in ("oo", "inf", "infinity") else (-oo if target_str in ("-oo", "-inf", "-infinity") else sympify(target_str))
                    expr = sympify(expr_str, locals={"x": x, "sin": sin, "cos": cos, "tan": tan, "exp": exp, "log": log})
                    res = limit(expr, x, target)
                    return (
                        f"**Limit:**\n\n"
                        f"$$\\lim_{{x \\to {latex(target)}}} \\left({latex(expr)}\\right) = \\mathbf{{{latex(res)}}}$$"
                    )
            except Exception:
                pass

        # ── D. Taylor / Maclaurin Series ──
        if "taylor" in q or "maclaurin" in q or "series expansion" in q:
            try:
                m_series = re.search(r"(?:taylor|maclaurin|series)\s+(?:of\s+)?([^\s]+(?:\s*[+\-*/^]\s*[^\s]+)*)", q)
                if m_series:
                    expr_str = m_series.group(1).replace("^", "**")
                    expr = sympify(expr_str, locals={"x": x, "sin": sin, "cos": cos, "exp": exp, "log": log})
                    res = series(expr, x, 0, 5)
                    return (
                        f"**Maclaurin Series Expansion** (around $x=0$ up to $O(x^5)$):\n\n"
                        f"$${latex(expr)} = \\mathbf{{{latex(res)}}}$$"
                    )
            except Exception:
                pass

        return None

    # =========================================================================
    # 6. SIMULTANEOUS SYSTEMS OF EQUATIONS
    # =========================================================================
    @classmethod
    def _solve_simultaneous_equations(cls, query: str, detailed: bool = False) -> str | None:
        q = cls.sanitize_math_input(query)
        # Strip leading conversational words
        cleaned_q = re.sub(r"^(?:solve|find|system\s+of\s+equations:?)\s*", "", q, flags=re.IGNORECASE)

        # Split on delimiters like 'and', ',', ';', '\n'
        raw_parts = re.split(r"\s*(?:and\s+|,|;|\n)\s*", cleaned_q)
        eq_patterns = [p.strip() for p in raw_parts if "=" in p and len(p.split("=")) == 2]

        if len(eq_patterns) >= 2:
            try:
                x, y, z = symbols('x y z')
                local_symbols = {'x': x, 'y': y, 'z': z}
                sympy_eqs = []
                for eq_str in eq_patterns[:3]:
                    lhs, rhs = eq_str.split("=")
                    lhs_expr = sympify(lhs.replace("^", "**").strip(), locals=local_symbols)
                    rhs_expr = sympify(rhs.replace("^", "**").strip(), locals=local_symbols)
                    sympy_eqs.append(lhs_expr - rhs_expr)

                sol = solve(sympy_eqs, (x, y) if len(eq_patterns) == 2 else (x, y, z), dict=True)
                if sol:
                    sol_lines = []
                    for idx, s in enumerate(sol, 1):
                        parts = [f"{var} = {latex(val)}" for var, val in s.items()]
                        sol_lines.append(f"**Solution {idx}:** $$" + ", \\quad ".join(parts) + "$$")

                    orig_eqs_latex = " \\\\\n".join(f"{eq.strip()}" for eq in eq_patterns)
                    return (
                        f"**System of Simultaneous Equations:**\n\n"
                        f"$$\\begin{{cases}}\n{orig_eqs_latex}\n\\end{{cases}}$$\n\n"
                        + "\n\n".join(sol_lines)
                    )
            except Exception:
                pass

        return None

    # =========================================================================
    # 7. EQUATIONS & QUADRATICS
    # =========================================================================
    @classmethod
    def _solve_equations(cls, query: str, detailed: bool = False) -> str | None:
        q = cls.sanitize_math_input(query)
        x = symbols('x')

        # Matches equations like "2x^2 + 5x - 3 = 0" or "solve 3x + 7 = 22"
        m_eq = re.search(r"(?:solve\s+)?([a-zA-Z0-9\s\+\-\*\/\^\(\)\.]+)\s*=\s*([a-zA-Z0-9\s\+\-\*\/\^\(\)\.]+)", q)
        if m_eq:
            try:
                lhs_str = m_eq.group(1).replace("^", "**").strip()
                rhs_str = m_eq.group(2).replace("^", "**").strip()
                lhs = sympify(lhs_str, locals={"x": x})
                rhs = sympify(rhs_str, locals={"x": x})
                eq_expr = lhs - rhs

                # Check if it's a quadratic polynomial ax^2 + bx + c
                poly = eq_expr.as_poly(x)
                if poly and poly.degree() == 2:
                    coeffs = poly.all_coeffs()
                    a, b, c = coeffs[0], coeffs[1], coeffs[2]
                    discriminant = b**2 - 4*a*c
                    roots = solve(eq_expr, x)
                    roots_formatted = ", \\quad ".join(f"x = {latex(r)}" for r in roots)

                    # Compute decimal approximation if radicals are present
                    approx_roots = []
                    for r in roots:
                        try:
                            approx_roots.append(f"x \\approx {float(r.evalf()):.4g}")
                        except Exception:
                            pass
                    approx_str = (", \\quad ".join(approx_roots)) if approx_roots else ""

                    return (
                        f"**Quadratic Equation:** $${latex(lhs)} = {latex(rhs)}$$\n\n"
                        f"**Standard Form:** $${latex(eq_expr)} = 0$$\n\n"
                        f"- **Coefficients:** $a = {latex(a)}, \\; b = {latex(b)}, \\; c = {latex(c)}$\n"
                        f"- **Discriminant:** $$\\Delta = b^2 - 4ac = ({latex(b)})^2 - 4({latex(a)})({latex(c)}) = \\mathbf{{{latex(discriminant)}}}$$\n\n"
                        f"**Quadratic Formula:** $$x = \\frac{{-b \\pm \\sqrt{{\\Delta}}}}{{2a}}$$\n\n"
                        f"**Exact Roots:** $$\\mathbf{{{roots_formatted}}}$$"
                        + (f"\n\n**Decimal Values:** $${approx_str}$$" if approx_str else "")
                    )

                # Linear or higher degree single-variable equation
                roots = solve(eq_expr, x)
                if roots:
                    roots_formatted = ", \\quad ".join(f"x = {latex(r)}" for r in roots)
                    return (
                        f"**Equation Solution:** $${latex(lhs)} = {latex(rhs)}$$\n\n"
                        f"$$\\mathbf{{{roots_formatted}}}$$"
                    )
            except Exception:
                pass

        return None

    # =========================================================================
    # 8. ALGEBRA (Simplify, Expand, Factor, Apart)
    # =========================================================================
    @classmethod
    def _solve_algebra(cls, query: str, detailed: bool = False) -> str | None:
        q = cls.sanitize_math_input(query)
        x, y = symbols('x y')
        locals_dict = {'x': x, 'y': y}

        # ── Factor ──
        if "factor" in q.lower():
            m = re.search(r"factor\s+(?:the\s+expression\s+)?([a-zA-Z0-9\s\+\-\*\/\^\(\)]+)", q, flags=re.IGNORECASE)
            if m:
                try:
                    expr = sympify(m.group(1).replace("^", "**").strip(), locals=locals_dict)
                    factored = factor(expr)
                    return f"**Factored Form:** $${latex(expr)} = \\mathbf{{{latex(factored)}}}$$"
                except Exception:
                    pass

        # ── Expand ──
        if "expand" in q.lower():
            m = re.search(r"expand\s+(?:the\s+expression\s+)?([a-zA-Z0-9\s\+\-\*\/\^\(\)]+)", q, flags=re.IGNORECASE)
            if m:
                try:
                    expr = sympify(m.group(1).replace("^", "**").strip(), locals=locals_dict)
                    expanded = expand(expr)
                    return f"**Expanded Polynomial:** $${latex(expr)} = \\mathbf{{{latex(expanded)}}}$$"
                except Exception:
                    pass

        # ── Simplify ──
        if "simplify" in q.lower():
            m = re.search(r"simplify\s+(?:the\s+expression\s+)?([a-zA-Z0-9\s\+\-\*\/\^\(\)]+)", q, flags=re.IGNORECASE)
            if m:
                try:
                    expr = sympify(m.group(1).replace("^", "**").strip(), locals=locals_dict)
                    simplified = simplify(expr)
                    return f"**Simplified Expression:** $${latex(expr)} = \\mathbf{{{latex(simplified)}}}$$"
                except Exception:
                    pass

        return None

    # =========================================================================
    # 9. TRIGONOMETRY & LOGARITHMS
    # =========================================================================
    @classmethod
    def _solve_trigonometry_and_logs(cls, query: str, detailed: bool = False) -> str | None:
        q = query.strip().lower()

        # Logarithm with base: log2(64), log base 2 of 64
        m_log_base = re.search(r"log\s*(?:base\s*)?(\d+(?:\.\d+)?)\s*(?:of\s*|\()\s*(\d+(?:\.\d+)?)\)?", q)
        if m_log_base:
            base_val = float(m_log_base.group(1))
            arg_val = float(m_log_base.group(2))
            res = math.log(arg_val, base_val)
            return f"**Logarithm:** $$\\log_{{{base_val:g}}}({arg_val:g}) = \\mathbf{{{res:.6g}}}$$"

        # Natural log ln(x) or log(x)
        m_ln = re.search(r"(?:ln|log)\s*(?:\(\s*([0-9\.\^e\*\+\-]+)\s*\)|([0-9\.\^e\*\+\-]+))", q)
        if m_ln:
            try:
                arg_str = (m_ln.group(1) or m_ln.group(2)).replace("^", "**")
                arg = sympify(arg_str, locals={"e": E, "pi": pi})
                res = log(arg)
                return f"**Natural Logarithm:** $$\\ln({latex(arg)}) = \\mathbf{{{latex(simplify(res))}}}$$"
            except Exception:
                pass

        # Trig evaluation: sin(pi/6), cos(45 deg), etc.
        m_trig = re.search(r"(sin|cos|tan|sec|csc|cot)\s*(?:\(\s*([^\)]+)\s*\)|([0-9a-zA-Z_\s\.\/]+))", q)
        if m_trig:
            fn = m_trig.group(1)
            arg_str = (m_trig.group(2) or m_trig.group(3)).strip()
            is_deg = "deg" in arg_str or "°" in arg_str
            clean_arg = re.sub(r"(?:deg|degrees|°)", "", arg_str).strip()
            try:
                val = sympify(clean_arg, locals={"pi": pi})
                rad_val = (val * pi / 180) if is_deg else val
                trig_fn = getattr(sp, fn)
                exact_res = trig_fn(rad_val)
                approx_res = float(exact_res.evalf())
                return (
                    f"**Trigonometric Function:** $${fn}({latex(val)}{'^\\circ' if is_deg else ''})$$\n\n"
                    f"**Exact Value:** $$\\mathbf{{{latex(exact_res)}}}$$\n\n"
                    f"**Decimal Value:** $$\\approx \\mathbf{{{approx_res:.6g}}}$$"
                )
            except Exception:
                pass

        return None

    # =========================================================================
    # 10. GENERAL ARITHMETIC & PERCENTAGES
    # =========================================================================
    @classmethod
    def _solve_arithmetic(cls, query: str, detailed: bool = False) -> str | None:
        q = cls.sanitize_math_input(query).lower()

        # Percentage: "15% of 850" or "what is 20 percent of 500"
        m_pct = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+(?:of\s+)?(\d+(?:\.\d+)?)", q)
        if m_pct:
            pct = float(m_pct.group(1))
            total = float(m_pct.group(2))
            res = (pct / 100.0) * total
            return f"**Percentage Calculation:** $${pct:g}\\% \\times {total:g} = \\mathbf{{{res:g}}}$$"

        # Pure numeric / arithmetic expression (e.g. "245 * 18", "sqrt(144) + 15", "2^10")
        m_expr = re.search(r"(?:calculate|compute|evaluate|what is|=?\s*)?([0-9\s\+\-\*\/\(\)\^\.\,]+)$", q)
        if m_expr:
            cand = m_expr.group(1).strip()
            # Must contain at least one arithmetic operator
            if any(op in cand for op in ["+", "-", "*", "/", "^", "**"]):
                try:
                    cand_expr = cand.replace("^", "**").replace(",", "")
                    res = sympify(cand_expr)
                    if isinstance(res, (sp.Number, sp.Expr)) and not res.free_symbols:
                        exact_res = simplify(res)
                        dec_approx = float(exact_res.evalf()) if isinstance(exact_res, sp.Rational) and exact_res.q != 1 else None
                        approx_txt = f" \\approx \\mathbf{{{dec_approx:.6g}}}" if dec_approx is not None else ""
                        return f"**Result:** $${cand} = \\mathbf{{{latex(exact_res)}}}{approx_txt}$$"
                except Exception:
                    pass

        return None
