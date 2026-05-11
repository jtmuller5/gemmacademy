package com.gemmacademy.student

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalContext
import com.gemmacademy.student.model.ModelStorage
import com.gemmacademy.student.ui.ChatScreen
import com.gemmacademy.student.ui.OnboardingScreen
import com.gemmacademy.student.ui.theme.GemmacademyTheme

sealed interface AppState {
    data object NeedsModel : AppState
    data object Ready : AppState
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            GemmacademyTheme {
                AppRoot()
            }
        }
    }
}

@Composable
private fun AppRoot() {
    val context = LocalContext.current
    var state by remember {
        mutableStateOf<AppState>(
            if (ModelStorage.isModelPresent(context)) AppState.Ready else AppState.NeedsModel
        )
    }

    when (state) {
        AppState.NeedsModel -> OnboardingScreen(onModelReady = { state = AppState.Ready })
        AppState.Ready -> ChatScreen()
    }
}
