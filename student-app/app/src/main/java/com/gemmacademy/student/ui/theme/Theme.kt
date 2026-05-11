package com.gemmacademy.student.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.sp

private val Teal = Color(0xFF0E6B73)
private val TealDark = Color(0xFF064951)
private val Cream = Color(0xFFFBF8F3)
private val Sand = Color(0xFFEFE7DA)
private val InkDark = Color(0xFF14202B)

private val LightColors = lightColorScheme(
    primary = Teal,
    onPrimary = Color.White,
    primaryContainer = Color(0xFFB5E5E8),
    onPrimaryContainer = TealDark,
    secondary = Color(0xFFC97A4A),
    onSecondary = Color.White,
    background = Cream,
    onBackground = InkDark,
    surface = Color.White,
    onSurface = InkDark,
    surfaceVariant = Sand,
    onSurfaceVariant = Color(0xFF40484E),
    outline = Color(0xFFB6BFC4),
)

private val AppTypography = Typography(
    bodyLarge = TextStyle(fontSize = 17.sp, lineHeight = 24.sp),
    bodyMedium = TextStyle(fontSize = 16.sp, lineHeight = 22.sp),
)

@Composable
fun GemmacademyTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = LightColors,
        typography = AppTypography,
        content = content,
    )
}
