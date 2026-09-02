#Requires -Version 5.1
# fold_router_chat.ps1 -- FoldRouter chat demo with left sidebar
# Implements the three-model routing table as a WPF chat UI.
#
# Layout: [icon bar 48px] | [chat history 200px] | [chat area *]
#
# Default routing (mirrors AdaptiveContentModelAdapter.cs FoldRouter):
#   Pop / Wo / Xul  ->  Gemma 3 1B    (language / creative / entropy)
#   Yax / Sek / Chen -> GPT-2 Large   (grammar / geodesic / projection)
#   AST mode (toggle): Sek + Wo -> Qwen  (coder / AST specialist override)
#
# Launch: pwsh -STA -File .\fold_router_chat.ps1

Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase, System.Windows.Forms

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------
$script:AstMode          = $false
$script:ActiveFold       = "Pop"
$script:ActiveModel      = "Gemma"
$script:Endpoint         = $null
$script:EndpointKind     = $null
$script:MaxTokens        = 512
$script:Temperature      = 0.7
$script:Conversation     = [System.Collections.ArrayList]::new()
$script:Sessions         = [System.Collections.ArrayList]::new()
$script:SessionMessages  = @{}
$script:CurrentSessionId = [System.Guid]::NewGuid().ToString("N").Substring(0,8)

$script:FoldColors = @{
    Pop  = "#58A6FF"
    Wo   = "#3FB950"
    Yax  = "#D29922"
    Sek  = "#F0883E"
    Chen = "#A78BFA"
    Xul  = "#56D4DD"
}
$script:ModelColors = @{
    Gemma     = "#238636"
    Gpt2Large = "#1F6FEB"
    Qwen      = "#DA3633"
}
$script:ModelLabels = @{
    Gemma     = "Gemma 3 1B"
    Gpt2Large = "GPT-2 Large"
    Qwen      = "Qwen 0.5B"
}

# Read live endpoints from active-model.json / chat.manifest.json written by START-SERVERS.bat
function Read-LiveEndpoints {
    $root = $PSScriptRoot
    $cmPath = Join-Path $root 'chat.manifest.json'
    $amPath = Join-Path $root 'active-model.json'
    $candidates = [System.Collections.ArrayList]::new()

    if (Test-Path $cmPath) {
        try {
            $cm = Get-Content $cmPath -Raw | ConvertFrom-Json
            if ($cm.gateway.chat) {
                $base = $cm.gateway.chat -replace '/v1/chat/completions',''
                $null = $candidates.Add(@{ Url = $base; Kind = "kuhul-gateway"; Health = "/v1/models" })
            }
            foreach ($url in $cm.fallback_chain) {
                $base = $url -replace '/v1/chat/completions',''
                if ($candidates | Where-Object { $_.Url -eq $base }) { continue }
                $null = $candidates.Add(@{ Url = $base; Kind = "llama"; Health = "/v1/models" })
            }
        } catch {}
    } elseif (Test-Path $amPath) {
        try {
            $am   = Get-Content $amPath -Raw | ConvertFrom-Json
            $base = $am.endpoint -replace '/v1/chat/completions',''
            $null = $candidates.Add(@{ Url = $base; Kind = "llama"; Health = "/v1/models" })
        } catch {}
    }

    foreach ($entry in @(
        @{ Url = "http://127.0.0.1:8764";  Kind = "kuhul-gateway"; Health = "/v1/models"   }
        @{ Url = "http://127.0.0.1:9000";  Kind = "llama";         Health = "/v1/models"   }
        @{ Url = "http://127.0.0.1:25110"; Kind = "gc-1";          Health = "/v1/models"   }
        @{ Url = "http://127.0.0.1:17480"; Kind = "kuhul-engine";  Health = "/health"      }
        @{ Url = "http://127.0.0.1:8787";  Kind = "json_runtime";  Health = "/api/health"  }
        @{ Url = "http://127.0.0.1:25501"; Kind = "mm-gemma-env";  Health = "/v1/models"   }
        @{ Url = "http://127.0.0.1:9003";  Kind = "dolphin";       Health = "/v1/models"   }
    )) {
        if (-not ($candidates | Where-Object { $_.Url -eq $entry.Url })) {
            $null = $candidates.Add($entry)
        }
    }
    return $candidates
}

$script:EndpointCandidates = Read-LiveEndpoints

# ---------------------------------------------------------------------------
#  XAML
# ---------------------------------------------------------------------------
[xml]$xaml = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="FoldRouter Chat"
        Height="700" Width="960" MinWidth="640" MinHeight="440"
        WindowStartupLocation="CenterScreen"
        Background="#0D1117" FontFamily="Consolas">
  <Window.Resources>
    <SolidColorBrush x:Key="{x:Static SystemColors.WindowBrushKey}"        Color="#0D1117"/>
    <SolidColorBrush x:Key="{x:Static SystemColors.WindowTextBrushKey}"    Color="#E6EDF3"/>
    <SolidColorBrush x:Key="{x:Static SystemColors.HighlightBrushKey}"     Color="#1F6FEB"/>
    <SolidColorBrush x:Key="{x:Static SystemColors.HighlightTextBrushKey}" Color="White"/>
    <SolidColorBrush x:Key="{x:Static SystemColors.ControlBrushKey}"       Color="#161B22"/>
    <SolidColorBrush x:Key="{x:Static SystemColors.ControlTextBrushKey}"   Color="#E6EDF3"/>
  </Window.Resources>

  <Grid>
    <Grid.ColumnDefinitions>
      <ColumnDefinition Width="48"/>    <!-- icon bar -->
      <ColumnDefinition Width="200"/>   <!-- history panel -->
      <ColumnDefinition Width="*"/>     <!-- chat area -->
    </Grid.ColumnDefinitions>
    <Grid.RowDefinitions>
      <RowDefinition Height="52"/>  <!-- header -->
      <RowDefinition Height="*"/>   <!-- content row -->
      <RowDefinition Height="62"/>  <!-- input bar -->
    </Grid.RowDefinitions>

    <!-- ── Header (spans all columns) ─────────────────────────────────── -->
    <Border Grid.Column="0" Grid.ColumnSpan="3" Grid.Row="0"
            Background="#161B22" BorderBrush="#21262D" BorderThickness="0,0,0,1">
      <DockPanel Margin="14,0">
        <!-- Right controls -->
        <StackPanel Orientation="Horizontal" DockPanel.Dock="Right" VerticalAlignment="Center">
          <ToggleButton x:Name="AstToggle" Content="AST mode" IsChecked="False"
                        Height="28" Padding="10,0" Margin="0,0,10,0" Cursor="Hand"
                        Background="#21262D" Foreground="#E6EDF3" BorderBrush="#30363D"
                        FontSize="11"
                        ToolTip="AST mode: Sek+Wo route to Qwen (coder / AST specialist)"/>
          <Border x:Name="ModelBadge" Background="#238636" CornerRadius="4"
                  Padding="8,3" Margin="0,0,10,0" VerticalAlignment="Center">
            <TextBlock x:Name="ModelBadgeText" Text="Gemma 3 1B"
                       Foreground="White" FontSize="11"/>
          </Border>
          <Border x:Name="FoldBadge" Background="#58A6FF" CornerRadius="4"
                  Padding="8,3" Margin="0,0,10,0" VerticalAlignment="Center">
            <TextBlock x:Name="FoldBadgeText" Text="Pop"
                       Foreground="White" FontSize="11" FontWeight="Bold"/>
          </Border>
          <Ellipse x:Name="StatusDot" Width="9" Height="9" Fill="#6E7681"
                   VerticalAlignment="Center" Margin="0,0,4,0"/>
          <TextBlock x:Name="EndpointText" Text="no endpoint"
                     Foreground="#6E7681" FontSize="10" VerticalAlignment="Center"/>
        </StackPanel>
        <!-- Left title -->
        <StackPanel Orientation="Horizontal" VerticalAlignment="Center">
          <TextBlock Text="FoldRouter Chat" Foreground="#E6EDF3"
                     FontSize="14" FontWeight="Bold" VerticalAlignment="Center"/>
          <TextBlock Text="  three-model routing demo"
                     Foreground="#6E7681" FontSize="11" VerticalAlignment="Center"/>
        </StackPanel>
      </DockPanel>
    </Border>

    <!-- ── Icon bar (col 0, rows 1+2) ─────────────────────────────────── -->
    <Border Grid.Column="0" Grid.Row="1" Grid.RowSpan="2"
            Background="#161B22" BorderBrush="#21262D" BorderThickness="0,0,1,0">
      <Grid>
        <Grid.RowDefinitions>
          <RowDefinition Height="*"/>
          <RowDefinition Height="Auto"/>
        </Grid.RowDefinitions>

        <!-- New chat button at top -->
        <StackPanel Grid.Row="0" VerticalAlignment="Top" HorizontalAlignment="Center"
                    Margin="0,8,0,0">
          <Button x:Name="NewChatBtn" Content="+" Width="32" Height="32"
                  Background="#21262D" Foreground="#58A6FF" BorderThickness="0"
                  FontSize="18" FontWeight="Bold" Cursor="Hand"
                  ToolTip="New chat session"/>
        </StackPanel>

        <!-- Profile button at bottom -->
        <Button x:Name="ProfileBtn" Grid.Row="1"
                Width="32" Height="32" Margin="0,0,0,10"
                HorizontalAlignment="Center" VerticalAlignment="Bottom"
                Background="#21262D" Foreground="#8B949E" BorderThickness="0"
                FontSize="14" Cursor="Hand"
                ToolTip="Session profile &amp; endpoint info"
                Content="&#x1F464;"/>
      </Grid>
    </Border>

    <!-- ── History panel (col 1, rows 1+2) ────────────────────────────── -->
    <Border Grid.Column="1" Grid.Row="1" Grid.RowSpan="2"
            Background="#0D1117" BorderBrush="#21262D" BorderThickness="0,0,1,0">
      <Grid>
        <Grid.RowDefinitions>
          <RowDefinition Height="Auto"/>
          <RowDefinition Height="*"/>
        </Grid.RowDefinitions>

        <!-- Panel header -->
        <Border Grid.Row="0" Background="#161B22" BorderBrush="#21262D"
                BorderThickness="0,0,0,1" Padding="10,8">
          <TextBlock Text="Chats" Foreground="#8B949E" FontSize="11" FontWeight="SemiBold"/>
        </Border>

        <!-- Session list -->
        <ScrollViewer Grid.Row="1" VerticalScrollBarVisibility="Auto"
                      HorizontalScrollBarVisibility="Disabled">
          <StackPanel x:Name="HistoryList" Margin="0,4,0,4"/>
        </ScrollViewer>
      </Grid>
    </Border>

    <!-- ── Chat feed (col 2, row 1) ───────────────────────────────────── -->
    <ScrollViewer Grid.Column="2" Grid.Row="1" x:Name="FeedScroll"
                  VerticalScrollBarVisibility="Auto" Background="#0D1117">
      <StackPanel x:Name="Feed" Margin="14,8,14,8"/>
    </ScrollViewer>

    <!-- ── Input bar (col 2, row 2) ───────────────────────────────────── -->
    <Border Grid.Column="2" Grid.Row="2"
            Background="#161B22" BorderBrush="#21262D" BorderThickness="0,1,0,0">
      <Grid Margin="10,10">
        <Grid.ColumnDefinitions>
          <ColumnDefinition Width="*"/>
          <ColumnDefinition Width="Auto"/>
          <ColumnDefinition Width="Auto"/>
          <ColumnDefinition Width="Auto"/>
        </Grid.ColumnDefinitions>
        <TextBox x:Name="InputBox"
                 Background="#0D1117" Foreground="#E6EDF3" CaretBrush="#58A6FF"
                 BorderBrush="#30363D" BorderThickness="1"
                 FontSize="13" Padding="10,6" Height="40" VerticalContentAlignment="Center"
                 ToolTip="Type a message -- fold phase and model are detected automatically"/>
        <Button x:Name="ReconnectBtn" Grid.Column="1" Content="Reconnect"
                Background="#21262D" Foreground="#58A6FF" BorderThickness="0"
                FontSize="11" Height="40" Padding="10,0" Margin="8,0,0,0" Cursor="Hand"
                ToolTip="Re-probe endpoints (run START-SERVERS.bat first)"/>
        <Button x:Name="ClearBtn" Grid.Column="2" Content="Clear"
                Background="#21262D" Foreground="#8B949E" BorderThickness="0"
                FontSize="11" Height="40" Padding="10,0" Margin="8,0,0,0" Cursor="Hand"
                ToolTip="Clear chat feed"/>
        <Button x:Name="SendBtn" Grid.Column="3" Content="  Send  "
                Background="#238636" Foreground="White" BorderThickness="0"
                FontSize="13" FontWeight="SemiBold" Height="40"
                Margin="8,0,0,0" Cursor="Hand"/>
      </Grid>
    </Border>
  </Grid>
</Window>
"@

$reader      = [System.Xml.XmlNodeReader]::new($xaml)
$window      = [System.Windows.Markup.XamlReader]::Load($reader)

$feed           = $window.FindName('Feed')
$scroll         = $window.FindName('FeedScroll')
$inputBox       = $window.FindName('InputBox')
$sendBtn        = $window.FindName('SendBtn')
$clearBtn       = $window.FindName('ClearBtn')
$reconnectBtn   = $window.FindName('ReconnectBtn')
$statusDot      = $window.FindName('StatusDot')
$endpointText   = $window.FindName('EndpointText')
$foldBadge      = $window.FindName('FoldBadge')
$foldBadgeText  = $window.FindName('FoldBadgeText')
$modelBadge     = $window.FindName('ModelBadge')
$modelBadgeText = $window.FindName('ModelBadgeText')
$astToggle      = $window.FindName('AstToggle')
$historyList    = $window.FindName('HistoryList')
$newChatBtn     = $window.FindName('NewChatBtn')
$profileBtn     = $window.FindName('ProfileBtn')
$script:window  = $window

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
function MkBrush([string]$hex) {
    [System.Windows.Media.SolidColorBrush]::new(
        [System.Windows.Media.ColorConverter]::ConvertFromString($hex))
}

function Set-Status([string]$color, [string]$label) {
    $window.Dispatcher.Invoke({
        $statusDot.Fill        = MkBrush $color
        $endpointText.Text     = $label
        $endpointText.Foreground = MkBrush $color
    })
}

function Update-FoldBadge([string]$fold, [string]$model) {
    $fc = if ($script:FoldColors.ContainsKey($fold))   { $script:FoldColors[$fold]   } else { "#8B949E" }
    $mc = if ($script:ModelColors.ContainsKey($model)) { $script:ModelColors[$model] } else { "#8B949E" }
    $ml = if ($script:ModelLabels.ContainsKey($model)) { $script:ModelLabels[$model] } else { $model }
    $window.Dispatcher.Invoke({
        $foldBadge.Background      = MkBrush $fc
        $foldBadgeText.Text        = $fold
        $modelBadge.Background     = MkBrush $mc
        $modelBadgeText.Text       = $ml
    })
}

# ---------------------------------------------------------------------------
#  FoldRouter logic (mirrors FoldRouter.DefaultRouting)
# ---------------------------------------------------------------------------
$script:DefaultRouting = @{
    Pop  = "Gemma"
    Wo   = "Gemma"
    Yax  = "Gpt2Large"
    Sek  = "Gpt2Large"
    Chen = "Gpt2Large"
    Xul  = "Gemma"
}

function Get-FoldPhase([string]$msg) {
    $l = $msg.ToLower()
    if ($l -match '\b(ast|asx|parse|syntax|token|grammar|compile|ir)\b')                          { return "Sek"  }
    if ($l -match '\b(load|search|find|read|fetch|look|what|who|when|where|list|show|get)\b')      { return "Pop"  }
    if ($l -match '\b(build|create|write|code|implement|define|construct|make|generate|scaffold)\b') { return "Wo"  }
    if ($l -match '\b(plan|predict|analyze|compare|evaluate|why|explain|design|think|reason)\b')   { return "Yax"  }
    if ($l -match '\b(execute|run|transform|convert|apply|calculate|compute|dispatch|call)\b')      { return "Sek"  }
    if ($l -match '\b(review|reflect|improve|optimize|refactor|summarize|check|validate|test)\b')  { return "Chen" }
    if ($l -match '\b(save|export|done|finish|complete|store|archive|commit|push)\b')               { return "Xul"  }
    if ($l -match '^\s*(hello|hi|hey|yo|sup|howdy|greetings)\b')                                    { return "Pop"  }
    return "Sek"
}

function Get-FoldModel([string]$fold) {
    if ($script:AstMode -and ($fold -eq "Sek" -or $fold -eq "Wo")) { return "Qwen" }
    if ($script:DefaultRouting.ContainsKey($fold)) { return $script:DefaultRouting[$fold] }
    return "Gemma"
}

# ---------------------------------------------------------------------------
#  Bubble rendering
# ---------------------------------------------------------------------------
function Add-UserBubble([string]$text) {
    $window.Dispatcher.Invoke({
        $b = [System.Windows.Controls.Border]::new()
        $b.CornerRadius = [System.Windows.CornerRadius]::new(14)
        $b.Margin       = [System.Windows.Thickness]::new(0,4,0,0)
        $b.Padding      = [System.Windows.Thickness]::new(14,10,14,10)
        $b.MaxWidth     = 600
        $b.Background   = MkBrush '#1F6FEB'
        $b.HorizontalAlignment = 'Right'
        $tb = [System.Windows.Controls.TextBlock]::new()
        $tb.Text        = $text
        $tb.Foreground  = [System.Windows.Media.Brushes]::White
        $tb.FontSize    = 13
        $tb.TextWrapping = 'Wrap'
        $b.Child = $tb
        $feed.Children.Add($b)
        $scroll.ScrollToBottom()
    })
}

function Add-AIBubble([string]$text, [string]$fold, [string]$model, [string]$endKind) {
    $fc = if ($script:FoldColors.ContainsKey($fold))   { $script:FoldColors[$fold]   } else { "#8B949E" }
    $mc = if ($script:ModelColors.ContainsKey($model)) { $script:ModelColors[$model] } else { "#8B949E" }
    $ml = if ($script:ModelLabels.ContainsKey($model)) { $script:ModelLabels[$model] } else { $model }
    $window.Dispatcher.Invoke({
        $outer = [System.Windows.Controls.StackPanel]::new()
        $outer.Margin = [System.Windows.Thickness]::new(0,4,0,0)
        $outer.HorizontalAlignment = 'Left'

        $b = [System.Windows.Controls.Border]::new()
        $b.CornerRadius = [System.Windows.CornerRadius]::new(14)
        $b.Padding      = [System.Windows.Thickness]::new(14,10,14,10)
        $b.MaxWidth     = 600
        $b.Background   = MkBrush '#21262D'
        $b.HorizontalAlignment = 'Left'
        $tb = [System.Windows.Controls.TextBlock]::new()
        $tb.Text        = $text
        $tb.Foreground  = MkBrush '#E6EDF3'
        $tb.FontSize    = 13
        $tb.TextWrapping = 'Wrap'
        $b.Child = $tb
        $outer.Children.Add($b)

        $badge = [System.Windows.Controls.StackPanel]::new()
        $badge.Orientation = 'Horizontal'
        $badge.Margin = [System.Windows.Thickness]::new(4,3,0,0)

        $foldTag = [System.Windows.Controls.Border]::new()
        $foldTag.Background   = MkBrush $fc
        $foldTag.CornerRadius = [System.Windows.CornerRadius]::new(3)
        $foldTag.Padding      = [System.Windows.Thickness]::new(6,1,6,1)
        $foldTag.Margin       = [System.Windows.Thickness]::new(0,0,4,0)
        $foldTb = [System.Windows.Controls.TextBlock]::new()
        $foldTb.Text      = $fold
        $foldTb.Foreground = [System.Windows.Media.Brushes]::White
        $foldTb.FontSize  = 9
        $foldTag.Child    = $foldTb
        $badge.Children.Add($foldTag)

        $modelTag = [System.Windows.Controls.Border]::new()
        $modelTag.Background   = MkBrush $mc
        $modelTag.CornerRadius = [System.Windows.CornerRadius]::new(3)
        $modelTag.Padding      = [System.Windows.Thickness]::new(6,1,6,1)
        $modelTag.Margin       = [System.Windows.Thickness]::new(0,0,4,0)
        $modelTb = [System.Windows.Controls.TextBlock]::new()
        $modelTb.Text      = $ml
        $modelTb.Foreground = [System.Windows.Media.Brushes]::White
        $modelTb.FontSize  = 9
        $modelTag.Child    = $modelTb
        $badge.Children.Add($modelTag)

        if ($endKind) {
            $ekTb = [System.Windows.Controls.TextBlock]::new()
            $ekTb.Text            = "via $endKind"
            $ekTb.Foreground      = MkBrush '#484F58'
            $ekTb.FontSize        = 9
            $ekTb.VerticalAlignment = 'Center'
            $badge.Children.Add($ekTb)
        }
        $outer.Children.Add($badge)
        $feed.Children.Add($outer)
        $scroll.ScrollToBottom()
    })
}

function Add-SysMessage([string]$text) {
    $window.Dispatcher.Invoke({
        $tb = [System.Windows.Controls.TextBlock]::new()
        $tb.Text       = $text
        $tb.Foreground = MkBrush '#484F58'
        $tb.FontSize   = 10
        $tb.Margin     = [System.Windows.Thickness]::new(4,5,0,2)
        $feed.Children.Add($tb)
        $scroll.ScrollToBottom()
    })
}

# ---------------------------------------------------------------------------
#  Session management
# ---------------------------------------------------------------------------
function Add-HistoryEntry([string]$sessionId, [string]$preview) {
    $window.Dispatcher.Invoke({
        $btn = [System.Windows.Controls.Button]::new()
        $btn.Tag = $sessionId
        $btn.Background = MkBrush '#0D1117'
        $btn.Foreground = MkBrush '#8B949E'
        $btn.BorderThickness = [System.Windows.Thickness]::new(0)
        $btn.HorizontalContentAlignment = 'Left'
        $btn.Padding     = [System.Windows.Thickness]::new(10,8,10,8)
        $btn.Margin      = [System.Windows.Thickness]::new(0,1,0,0)
        $btn.Cursor      = [System.Windows.Input.Cursors]::Hand
        $btn.ToolTip     = $preview

        $sp = [System.Windows.Controls.StackPanel]::new()

        $dot = [System.Windows.Controls.TextBlock]::new()
        $short = if ($preview.Length -gt 26) { $preview.Substring(0,26) + "..." } else { $preview }
        $dot.Text        = $short
        $dot.FontSize    = 11
        $dot.TextWrapping = 'Wrap'
        $sp.Children.Add($dot)

        $idTb = [System.Windows.Controls.TextBlock]::new()
        $idTb.Text        = "#$sessionId"
        $idTb.FontSize    = 9
        $idTb.Foreground  = MkBrush '#484F58'
        $sp.Children.Add($idTb)

        $btn.Content = $sp
        $btn.Add_Click({
            $id = $_.Source.Tag
            Load-Session $id
        })
        $historyList.Children.Insert(0, $btn)
    })
}

function Save-CurrentSession {
    if ($script:Conversation.Count -eq 0) { return }
    $copy = [System.Collections.ArrayList]::new()
    foreach ($m in $script:Conversation) { $null = $copy.Add($m) }
    $script:SessionMessages[$script:CurrentSessionId] = $copy
}

function New-Session {
    Save-CurrentSession
    # Mark current history entry as inactive (dim it, skip recolor for simplicity)
    $script:CurrentSessionId = [System.Guid]::NewGuid().ToString("N").Substring(0,8)
    $null = $script:Sessions.Add($script:CurrentSessionId)
    $script:Conversation.Clear()
    $window.Dispatcher.Invoke({ $feed.Children.Clear() })
    Add-SysMessage "New session: #$($script:CurrentSessionId)"
    Add-SysMessage "Routing: Pop/Wo/Xul->Gemma | Yax/Sek/Chen->GPT-2 Large | AST->Qwen"
}

function Load-Session([string]$sessionId) {
    if (-not $script:SessionMessages.ContainsKey($sessionId)) {
        Add-SysMessage "Session #$sessionId not found."
        return
    }
    Save-CurrentSession
    $script:CurrentSessionId = $sessionId
    $script:Conversation.Clear()
    foreach ($m in $script:SessionMessages[$sessionId]) {
        $null = $script:Conversation.Add($m)
    }
    $window.Dispatcher.Invoke({ $feed.Children.Clear() })
    Add-SysMessage "Loaded session #$sessionId"
    foreach ($m in $script:Conversation) {
        if ($m.role -eq 'user')      { Add-UserBubble $m.content }
        elseif ($m.role -eq 'assistant') { Add-AIBubble $m.content "Sek" "Gemma" "" }
    }
}

# ---------------------------------------------------------------------------
#  Profile modal
# ---------------------------------------------------------------------------
function Show-ProfileModal {
    $modal = [System.Windows.Window]::new()
    $modal.Title   = "Session Profile"
    $modal.Width   = 360
    $modal.Height  = 300
    $modal.ResizeMode = 'NoResize'
    $modal.WindowStartupLocation = 'CenterOwner'
    $modal.Owner   = $script:window
    $modal.Background = MkBrush '#0D1117'
    $modal.FontFamily = [System.Windows.Media.FontFamily]::new("Consolas")

    $sp = [System.Windows.Controls.StackPanel]::new()
    $sp.Margin = [System.Windows.Thickness]::new(22,18,22,18)

    $title = [System.Windows.Controls.TextBlock]::new()
    $title.Text       = "FoldRouter Session"
    $title.Foreground = MkBrush '#E6EDF3'
    $title.FontSize   = 15
    $title.FontWeight = 'SemiBold'
    $title.Margin     = [System.Windows.Thickness]::new(0,0,0,14)
    $sp.Children.Add($title)

    $astLabel = if ($script:AstMode) { "ON  (Sek/Wo -> Qwen)" } else { "OFF (default routing)" }
    $rows = @(
        @("Session ID",    "#$($script:CurrentSessionId)")
        @("Sessions",      "$($script:Sessions.Count + 1) total")
        @("Endpoint",      ($script:Endpoint    ?? "(none)"))
        @("Kind",          ($script:EndpointKind ?? "(offline)"))
        @("AST mode",      $astLabel)
        @("Conversation",  "$($script:Conversation.Count) entries")
    )

    foreach ($row in $rows) {
        $g = [System.Windows.Controls.Grid]::new()
        $g.Margin = [System.Windows.Thickness]::new(0,4,0,4)
        $c0 = [System.Windows.Controls.ColumnDefinition]::new()
        $c0.Width = [System.Windows.GridLength]::new(110)
        $c1 = [System.Windows.Controls.ColumnDefinition]::new()
        $c1.Width = [System.Windows.GridLength]::new(1, [System.Windows.GridUnitType]::Star)
        $g.ColumnDefinitions.Add($c0)
        $g.ColumnDefinitions.Add($c1)

        $lbl = [System.Windows.Controls.TextBlock]::new()
        $lbl.Text       = $row[0] + ":"
        $lbl.Foreground = MkBrush '#8B949E'
        $lbl.FontSize   = 12
        [System.Windows.Controls.Grid]::SetColumn($lbl, 0)
        $g.Children.Add($lbl)

        $val = [System.Windows.Controls.TextBlock]::new()
        $val.Text        = $row[1]
        $val.Foreground  = MkBrush '#E6EDF3'
        $val.FontSize    = 12
        $val.TextWrapping = 'Wrap'
        [System.Windows.Controls.Grid]::SetColumn($val, 1)
        $g.Children.Add($val)

        $sp.Children.Add($g)
    }

    # Routing table summary
    $sep = [System.Windows.Controls.TextBlock]::new()
    $sep.Text       = "`nRouting: Pop/Wo/Xul=Gemma  Yax/Sek/Chen=GPT-2L  AST=Qwen"
    $sep.Foreground = MkBrush '#484F58'
    $sep.FontSize   = 10
    $sep.TextWrapping = 'Wrap'
    $sp.Children.Add($sep)

    $modal.Content = $sp
    $null = $modal.ShowDialog()
}

# ---------------------------------------------------------------------------
#  Endpoint probing
# ---------------------------------------------------------------------------
function Find-Endpoint {
    $script:EndpointCandidates = Read-LiveEndpoints
    foreach ($c in $script:EndpointCandidates) {
        try {
            $null = Invoke-RestMethod -Uri "$($c.Url)$($c.Health)" -Method Get -TimeoutSec 2 -ErrorAction Stop
            $script:Endpoint     = $c.Url
            $script:EndpointKind = $c.Kind
            $port = ([Uri]$c.Url).Port
            Set-Status '#3FB950' "$($c.Kind) :$port"
            Add-SysMessage "Connected: $($c.Kind) at $($c.Url)"
            return $true
        } catch { }
    }
    Set-Status '#F0883E' "offline -- responses simulated"
    Add-SysMessage "No live endpoint found. Run START-SERVERS.bat then click Reconnect."
    return $false
}

# ---------------------------------------------------------------------------
#  Inference dispatch
# ---------------------------------------------------------------------------
function Invoke-FoldInference([string]$prompt, [string]$fold, [string]$model) {
    if (-not $script:Endpoint) { return $null }

    $msgs = [System.Collections.ArrayList]::new()
    foreach ($m in $script:Conversation) { $null = $msgs.Add($m) }
    $null = $msgs.Add(@{ role = "user"; content = $prompt })

    if ($script:EndpointKind -eq "json_runtime") {
        $body = @{
            "@program" = "fold_router.kuhul"
            "request"  = @{
                "prompt"      = $prompt
                "fold"        = $fold
                "model"       = $model
                "max_tokens"  = $script:MaxTokens
                "temperature" = $script:Temperature
                "messages"    = @($msgs)
            }
        }
        try {
            $resp = Invoke-RestMethod -Uri "$($script:Endpoint)/api/run" `
                -Method Post `
                -Body ($body | ConvertTo-Json -Depth 8 -Compress) `
                -ContentType "application/json" `
                -TimeoutSec 120
            if ($resp.response) { return [string]$resp.response }
            if ($resp.content)  { return [string]$resp.content  }
        } catch { return $null }
        return $null
    }

    $body = @{
        model       = "gemma-3-1b-it-q8_0"
        messages    = @($msgs)
        max_tokens  = $script:MaxTokens
        temperature = $script:Temperature
        stream      = $false
    }
    try {
        $resp = Invoke-RestMethod -Uri "$($script:Endpoint)/v1/chat/completions" `
            -Method Post `
            -Body ($body | ConvertTo-Json -Depth 8 -Compress) `
            -ContentType "application/json" `
            -TimeoutSec 120
        $content = $resp.choices[0].message.content
        if ($content) { return [string]$content }
    } catch { return $null }
    return $null
}

function Get-SimulatedResponse([string]$prompt, [string]$fold, [string]$model) {
    $ml = if ($script:ModelLabels.ContainsKey($model)) { $script:ModelLabels[$model] } else { $model }
    return "(offline) Fold: $fold -- Model: $ml -- Prompt: $($prompt.Substring(0,[Math]::Min(80,$prompt.Length)))..."
}

# ---------------------------------------------------------------------------
#  Send handler
# ---------------------------------------------------------------------------
function Send-Message([string]$msg) {
    $msg = $msg.Trim()
    if ([string]::IsNullOrWhiteSpace($msg)) { return }

    # First message in session -> add history entry
    if ($script:Conversation.Count -eq 0) {
        $preview = if ($msg.Length -gt 40) { $msg.Substring(0,40) + "..." } else { $msg }
        Add-HistoryEntry $script:CurrentSessionId $preview
        $null = $script:Sessions.Add($script:CurrentSessionId)
    }

    Add-UserBubble $msg

    $fold  = Get-FoldPhase $msg
    $model = Get-FoldModel $fold
    $script:ActiveFold  = $fold
    $script:ActiveModel = $model
    Update-FoldBadge $fold $model

    Set-Status '#F0883E' "$($script:EndpointKind) thinking..."
    [System.Windows.Forms.Application]::DoEvents()

    $response = Invoke-FoldInference $msg $fold $model
    if (-not $response) { $response = Get-SimulatedResponse $msg $fold $model }
    if ([string]::IsNullOrWhiteSpace($response)) { $response = "(empty response)" }

    Add-AIBubble $response $fold $model $script:EndpointKind

    $null = $script:Conversation.Add(@{ role = "user";      content = $msg      })
    $null = $script:Conversation.Add(@{ role = "assistant"; content = $response })
    while ($script:Conversation.Count -gt 40) { $script:Conversation.RemoveAt(0) }

    if ($script:Endpoint) {
        Set-Status '#3FB950' "$($script:EndpointKind) $($script:Endpoint)"
    } else {
        Set-Status '#F0883E' "offline"
    }
}

# ---------------------------------------------------------------------------
#  Event wiring
# ---------------------------------------------------------------------------
$sendBtn.Add_Click({
    $msg = $inputBox.Text
    $inputBox.Text = ""
    Send-Message $msg
})

$inputBox.Add_KeyDown({
    if ($_.Key -eq 'Enter' -and -not $_.KeyboardDevice.Modifiers) {
        $msg = $inputBox.Text
        $inputBox.Text = ""
        Send-Message $msg
        $_.Handled = $true
    }
})

$reconnectBtn.Add_Click({
    $script:Endpoint     = $null
    $script:EndpointKind = $null
    Set-Status '#F0883E' "probing..."
    Add-SysMessage "Probing endpoints..."
    [System.Windows.Forms.Application]::DoEvents()
    $null = Find-Endpoint
})

$clearBtn.Add_Click({
    $feed.Children.Clear()
    $script:Conversation.Clear()
    Add-SysMessage "Chat cleared."
})

$astToggle.Add_Checked({
    $script:AstMode = $true
    Add-SysMessage "AST mode ON: Sek+Wo -> Qwen (coder / AST specialist)"
    Update-FoldBadge $script:ActiveFold (Get-FoldModel $script:ActiveFold)
})
$astToggle.Add_Unchecked({
    $script:AstMode = $false
    Add-SysMessage "AST mode OFF: default routing restored"
    Update-FoldBadge $script:ActiveFold (Get-FoldModel $script:ActiveFold)
})

$newChatBtn.Add_Click({
    New-Session
})

$profileBtn.Add_Click({
    Show-ProfileModal
})

# ---------------------------------------------------------------------------
#  Startup
# ---------------------------------------------------------------------------
Add-SysMessage "FoldRouter Chat -- three-model routing demo"
Add-SysMessage "Routing: Pop/Wo/Xul->Gemma | Yax/Sek/Chen->GPT-2 Large | AST toggle->Qwen"
Add-SysMessage "Probing endpoints..."

$null = Find-Endpoint

$window.Add_Loaded({
    Add-SysMessage "Routing table active. Type a message to test fold detection."
    Add-AIBubble "Hello. I'm the FoldRouter demo. Each message is classified to a fold phase and routed to the right model specialist. Try 'write code' (Wo->Gemma), 'analyze a plan' (Yax->GPT-2L), or enable AST mode to route code to Qwen." "Pop" "Gemma" ""
})

[void]$window.ShowDialog()
