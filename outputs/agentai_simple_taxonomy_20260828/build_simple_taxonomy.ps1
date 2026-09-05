param(
    [ValidateSet('PreviewSource', 'BuildDraft', 'VerifyFinal')]
    [string]$Mode = 'BuildDraft',
    [string]$InputPath = 'C:\Users\82108\Downloads\agentAI_주요사항보고서_분류체계_수정본.xlsx',
    [string]$OutputDir = 'C:\Users\82108\Miraeasset_AI_Festival\outputs\agentai_simple_taxonomy_20260828'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$outputPath = Join-Path $OutputDir 'agentAI_주요사항보고서_AI학습용_단순분류.xlsx'
$sourcePreviewPath = Join-Path $OutputDir 'preview_source.png'
$dataPreviewPath = Join-Path $OutputDir 'preview_ai_training_data.png'
$summaryPreviewPath = Join-Path $OutputDir 'preview_group_summary.png'

function Get-OleColor {
    param([int]$Red, [int]$Green, [int]$Blue)
    return $Red + (256 * $Green) + (65536 * $Blue)
}

function Release-ComObject {
    param([object]$ComObject)
    if ($null -ne $ComObject) {
        try { [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($ComObject) } catch { }
    }
}

function Export-WorksheetPng {
    param(
        [object]$Worksheet,
        [string]$Path
    )

    $range = $null
    $bitmap = $null
    try {
        $Worksheet.Activate()
        $range = $Worksheet.UsedRange
        $range.CopyPicture(1, 2)
        Start-Sleep -Milliseconds 900
        $bitmap = [System.Windows.Forms.Clipboard]::GetImage()
        if ($null -eq $bitmap) {
            throw "Excel did not place a worksheet image on the clipboard."
        }
        $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
        if (-not (Test-Path -LiteralPath $Path)) {
            throw "Worksheet preview export failed: $Path"
        }
    }
    finally {
        if ($null -ne $bitmap) { $bitmap.Dispose() }
        Release-ComObject $range
    }
}

function New-OneBasedMatrix {
    param([int]$Rows, [int]$Columns)
    $matrix = [Array]::CreateInstance([object], [int[]]@($Rows, $Columns), [int[]]@(1, 1))
    return ,$matrix
}

function Get-Mapping {
    $mapping = [ordered]@{}
    $mapping['주요사항보고서(자기주식처분결정)'] = [pscustomobject]@{ Group='자본·주식관리'; Detail='자기주식 처분'; Label='EQUITY_MANAGEMENT' }
    $mapping['주요사항보고서(자기주식취득결정)'] = [pscustomobject]@{ Group='자본·주식관리'; Detail='자기주식 취득'; Label='EQUITY_MANAGEMENT' }
    $mapping['주요사항보고서(상각형조건부자본증권발행결정)'] = [pscustomobject]@{ Group='자금조달·메자닌'; Detail='자본성·조건부자본증권 발행'; Label='FINANCING_SECURITIES' }
    $mapping['주요사항보고서(유상증자결정)'] = [pscustomobject]@{ Group='자금조달·메자닌'; Detail='유상증자'; Label='FINANCING_SECURITIES' }
    $mapping['주요사항보고서(자기주식취득신탁계약체결결정)'] = [pscustomobject]@{ Group='자본·주식관리'; Detail='자기주식 신탁'; Label='EQUITY_MANAGEMENT' }
    $mapping['주요사항보고서(자기주식취득신탁계약해지결정)'] = [pscustomobject]@{ Group='자본·주식관리'; Detail='자기주식 신탁'; Label='EQUITY_MANAGEMENT' }
    $mapping['주요사항보고서(회사합병결정)'] = [pscustomobject]@{ Group='기업결합·조직재편'; Detail='회사 합병'; Label='CORPORATE_RESTRUCTURING' }
    $mapping['주요사항보고서(전환사채권발행결정)'] = [pscustomobject]@{ Group='자금조달·메자닌'; Detail='전환사채(CB)'; Label='FINANCING_SECURITIES' }
    $mapping['주요사항보고서(주식교환·이전결정)'] = [pscustomobject]@{ Group='기업결합·조직재편'; Detail='주식교환·이전'; Label='CORPORATE_RESTRUCTURING' }
    $mapping['주요사항보고서(회사분할결정)'] = [pscustomobject]@{ Group='기업결합·조직재편'; Detail='기업 분할'; Label='CORPORATE_RESTRUCTURING' }
    $mapping['주요사항보고서(회사분할합병결정)'] = [pscustomobject]@{ Group='기업결합·조직재편'; Detail='기업 분할'; Label='CORPORATE_RESTRUCTURING' }
    $mapping['주요사항보고서(타법인주식및출자증권양수결정)'] = [pscustomobject]@{ Group='투자·자산거래'; Detail='타법인 지분 거래'; Label='INVESTMENT_ASSET_TRANSACTIONS' }
    $mapping['주요사항보고서(감자결정)'] = [pscustomobject]@{ Group='자본·주식관리'; Detail='감자'; Label='EQUITY_MANAGEMENT' }
    $mapping['주요사항보고서(자본으로인정되는채무증권발행결정)'] = [pscustomobject]@{ Group='자금조달·메자닌'; Detail='자본성·조건부자본증권 발행'; Label='FINANCING_SECURITIES' }
    $mapping['주요사항보고서(무상증자결정)'] = [pscustomobject]@{ Group='자본·주식관리'; Detail='무상증자'; Label='EQUITY_MANAGEMENT' }
    $mapping['주요사항보고서(소송등의제기)'] = [pscustomobject]@{ Group='경영위험'; Detail='소송·분쟁'; Label='BUSINESS_RISK' }
    $mapping['주요사항보고서(교환사채권발행결정)'] = [pscustomobject]@{ Group='자금조달·메자닌'; Detail='교환사채(EB)'; Label='FINANCING_SECURITIES' }
    $mapping['주요사항보고서(해외증권시장주권등상장폐지)'] = [pscustomobject]@{ Group='상장·시장변경'; Detail='해외 상장폐지'; Label='LISTING_STATUS' }
    $mapping['주요사항보고서(해외증권시장주권등상장폐지결정)'] = [pscustomobject]@{ Group='상장·시장변경'; Detail='해외 상장폐지'; Label='LISTING_STATUS' }
    $mapping['주요사항보고서(유형자산양도결정)'] = [pscustomobject]@{ Group='투자·자산거래'; Detail='유형자산 거래'; Label='INVESTMENT_ASSET_TRANSACTIONS' }
    $mapping['주요사항보고서(영업양수결정)'] = [pscustomobject]@{ Group='기업결합·조직재편'; Detail='영업 양수'; Label='CORPORATE_RESTRUCTURING' }
    $mapping['주요사항보고서(유형자산양수결정)'] = [pscustomobject]@{ Group='투자·자산거래'; Detail='유형자산 거래'; Label='INVESTMENT_ASSET_TRANSACTIONS' }
    $mapping['주요사항보고서(자기전환사채매도결정)'] = [pscustomobject]@{ Group='자금조달·메자닌'; Detail='전환사채(CB)'; Label='FINANCING_SECURITIES' }
    $mapping['주요사항보고서(영업정지)'] = [pscustomobject]@{ Group='경영위험'; Detail='영업정지'; Label='BUSINESS_RISK' }
    $mapping['주요사항보고서(제3자의전환사채매수선택권행사)'] = [pscustomobject]@{ Group='자금조달·메자닌'; Detail='전환사채(CB)'; Label='FINANCING_SECURITIES' }
    $mapping['주요사항보고서(타법인주식및출자증권양도결정)'] = [pscustomobject]@{ Group='투자·자산거래'; Detail='타법인 지분 거래'; Label='INVESTMENT_ASSET_TRANSACTIONS' }
    $mapping['주요사항보고서(해외증권시장주권등상장)'] = [pscustomobject]@{ Group='상장·시장변경'; Detail='해외 상장'; Label='LISTING_STATUS' }
    $mapping['주요사항보고서(해외증권시장주권등상장결정)'] = [pscustomobject]@{ Group='상장·시장변경'; Detail='해외 상장'; Label='LISTING_STATUS' }
    return $mapping
}

function Set-WindowStyle {
    param([object]$Excel, [object]$Worksheet, [int]$SplitRow)
    $Worksheet.Activate()
    $window = $Excel.ActiveWindow
    $window.DisplayGridlines = $false
    $window.SplitColumn = 0
    $window.SplitRow = $SplitRow
    $window.FreezePanes = $true
    Release-ComObject $window
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

if ($Mode -eq 'PreviewSource') {
    $excel = $null
    $workbook = $null
    $worksheet = $null
    try {
        $excel = New-Object -ComObject Excel.Application
        $excel.Visible = $false
        $excel.DisplayAlerts = $false
        $excel.ScreenUpdating = $true
        $workbook = $excel.Workbooks.Open($InputPath, 0, $true)
        $worksheet = $workbook.Worksheets.Item('공시 분류 데이터')
        Export-WorksheetPng -Worksheet $worksheet -Path $sourcePreviewPath
        [pscustomobject]@{ mode=$Mode; preview=$sourcePreviewPath; rows=$worksheet.UsedRange.Rows.Count; columns=$worksheet.UsedRange.Columns.Count } | ConvertTo-Json -Compress
    }
    finally {
        if ($null -ne $workbook) { try { $workbook.Close($false) } catch { } }
        if ($null -ne $excel) { try { $excel.Quit() } catch { } }
        Release-ComObject $worksheet
        Release-ComObject $workbook
        Release-ComObject $excel
        [GC]::Collect()
        [GC]::WaitForPendingFinalizers()
    }
    exit 0
}

if ($Mode -eq 'BuildDraft') {
    $mapping = Get-Mapping
    if ($mapping.Count -ne 28) { throw "Expected 28 mapping rules, found $($mapping.Count)." }
    Write-Output 'STEP 1 mapping_ready'

    $excel = $null
    $sourceWorkbook = $null
    $sourceWorksheet = $null
    $sourceRange = $null
    $workbook = $null
    $dataSheet = $null
    $summarySheet = $null
    $dataTable = $null
    try {
        $excel = New-Object -ComObject Excel.Application
        $excel.Visible = $false
        $excel.DisplayAlerts = $false
        $excel.ScreenUpdating = $true
        Write-Output 'STEP 2 excel_started'

        $sourceWorkbook = $excel.Workbooks.Open($InputPath, 0, $true)
        $sourceWorksheet = $sourceWorkbook.Worksheets.Item('공시 분류 데이터')
        $sourceRange = $sourceWorksheet.Range('A2:B29')
        $sourceValues = $sourceRange.Value2
        $sourceWorkbook.Close($false)
        Release-ComObject $sourceRange
        $sourceRange = $null
        Release-ComObject $sourceWorksheet
        $sourceWorksheet = $null
        Release-ComObject $sourceWorkbook
        $sourceWorkbook = $null
        Write-Output 'STEP 3 source_loaded'

        $dataMatrix = New-OneBasedMatrix -Rows 29 -Columns 5
        $headers = @('공시 제목', '문서 수', '통합 분류', '세부 유형', '학습용 라벨')
        for ($column = 1; $column -le 5; $column++) {
            $dataMatrix.SetValue($headers[$column - 1], 1, $column)
        }

        $usedTitles = [System.Collections.Generic.HashSet[string]]::new()
        $sourceRowLowerBound = $sourceValues.GetLowerBound(0)
        $sourceColumnLowerBound = $sourceValues.GetLowerBound(1)
        for ($index = 0; $index -lt 28; $index++) {
            $title = [string]$sourceValues.GetValue($sourceRowLowerBound + $index, $sourceColumnLowerBound)
            $documentCount = [int]$sourceValues.GetValue($sourceRowLowerBound + $index, $sourceColumnLowerBound + 1)
            if (-not $mapping.Contains($title)) { throw "No simplified mapping exists for: $title" }
            [void]$usedTitles.Add($title)
            $rule = $mapping[$title]
            $excelRow = $index + 2
            $dataMatrix.SetValue($title, $excelRow, 1)
            $dataMatrix.SetValue($documentCount, $excelRow, 2)
            $dataMatrix.SetValue($rule.Group, $excelRow, 3)
            $dataMatrix.SetValue($rule.Detail, $excelRow, 4)
            $dataMatrix.SetValue($rule.Label, $excelRow, 5)
        }
        if ($usedTitles.Count -ne 28) { throw "Expected 28 unique source titles, found $($usedTitles.Count)." }
        Write-Output 'STEP 4 data_matrix_ready'

        $workbook = $excel.Workbooks.Add()
        while ($workbook.Worksheets.Count -lt 2) { [void]$workbook.Worksheets.Add() }
        while ($workbook.Worksheets.Count -gt 2) { $workbook.Worksheets.Item($workbook.Worksheets.Count).Delete() }
        $dataSheet = $workbook.Worksheets.Item(1)
        $summarySheet = $workbook.Worksheets.Item(2)
        $dataSheet.Name = 'AI 학습용 분류'
        $summarySheet.Name = '그룹 기준'
        Write-Output 'STEP 5 workbook_ready'

        $dataSheet.Range('A1:E29').Value2 = $dataMatrix
        $dataSheet.Cells.Font.Name = '맑은 고딕'
        $dataSheet.Cells.Font.Size = 10
        $dataSheet.Range('A1:E1').Interior.Color = Get-OleColor 31 78 121
        $dataSheet.Range('A1:E1').Font.Color = Get-OleColor 255 255 255
        $dataSheet.Range('A1:E1').Font.Bold = $true
        $dataSheet.Range('A1:E1').HorizontalAlignment = -4108
        $dataSheet.Range('A1:E29').VerticalAlignment = -4108
        $dataSheet.Range('A2:A29').HorizontalAlignment = -4131
        $dataSheet.Range('B2:B29').HorizontalAlignment = -4152
        $dataSheet.Range('C2:C29').HorizontalAlignment = -4108
        $dataSheet.Range('D2:D29').HorizontalAlignment = -4131
        $dataSheet.Range('E2:E29').HorizontalAlignment = -4108
        $dataSheet.Range('B2:B29').NumberFormat = '#,##0'
        $dataSheet.Range('A1:E29').WrapText = $false
        $dataSheet.Columns.Item('A').ColumnWidth = 48
        $dataSheet.Columns.Item('B').ColumnWidth = 10
        $dataSheet.Columns.Item('C').ColumnWidth = 22
        $dataSheet.Columns.Item('D').ColumnWidth = 30
        $dataSheet.Columns.Item('E').ColumnWidth = 34
        $dataSheet.Rows.Item(1).RowHeight = 27
        $dataSheet.Range('2:29').RowHeight = 22
        $dataSheet.Tab.Color = Get-OleColor 31 78 121

        $groupColors = [ordered]@{
            '자본·주식관리' = Get-OleColor 221 235 247
            '자금조달·메자닌' = Get-OleColor 226 239 218
            '기업결합·조직재편' = Get-OleColor 252 228 214
            '투자·자산거래' = Get-OleColor 255 242 204
            '상장·시장변경' = Get-OleColor 228 223 236
            '경영위험' = Get-OleColor 244 204 204
        }
        for ($row = 2; $row -le 29; $row++) {
            $groupName = [string]$dataSheet.Cells.Item($row, 3).Value2
            $fillColor = $groupColors[$groupName]
            $dataSheet.Cells.Item($row, 3).Interior.Color = $fillColor
            $dataSheet.Cells.Item($row, 3).Font.Bold = $true
            $dataSheet.Cells.Item($row, 5).Interior.Color = $fillColor
        }

        $dataTable = $dataSheet.ListObjects.Add(1, $dataSheet.Range('A1:E29'), $null, 1)
        $dataTable.Name = 'SimplifiedDisclosureTaxonomy'
        $dataTable.TableStyle = 'TableStyleMedium2'
        Set-WindowStyle -Excel $excel -Worksheet $dataSheet -SplitRow 1
        Write-Output 'STEP 6 data_sheet_styled'

        $dataSheet.PageSetup.Orientation = 2
        $dataSheet.PageSetup.Zoom = $false
        $dataSheet.PageSetup.FitToPagesWide = 1
        $dataSheet.PageSetup.FitToPagesTall = 1
        $dataSheet.PageSetup.PrintArea = '$A$1:$E$29'
        $dataSheet.PageSetup.LeftMargin = $excel.InchesToPoints(0.25)
        $dataSheet.PageSetup.RightMargin = $excel.InchesToPoints(0.25)
        $dataSheet.PageSetup.TopMargin = $excel.InchesToPoints(0.4)
        $dataSheet.PageSetup.BottomMargin = $excel.InchesToPoints(0.4)

        $summarySheet.Cells.Font.Name = '맑은 고딕'
        $summarySheet.Cells.Font.Size = 10
        $summarySheet.Range('A1:F1').Merge()
        $summarySheet.Range('A1').Value2 = 'AI 학습용 6개 통합 분류'
        $summarySheet.Range('A1:F1').Interior.Color = Get-OleColor 31 78 121
        $summarySheet.Range('A1:F1').Font.Color = Get-OleColor 255 255 255
        $summarySheet.Range('A1:F1').Font.Bold = $true
        $summarySheet.Range('A1:F1').Font.Size = 16
        $summarySheet.Range('A1:F1').HorizontalAlignment = -4108
        $summarySheet.Range('A1:F1').VerticalAlignment = -4108
        $summarySheet.Rows.Item(1).RowHeight = 34
        $summarySheet.Range('A2:F2').Merge()
        $summarySheet.Range('A2').Value2 = '공시 제목만으로 구분하기 쉬운 경제적 사건군으로 단순화했습니다.'
        $summarySheet.Range('A2:F2').Font.Color = Get-OleColor 89 89 89
        $summarySheet.Range('A2:F2').HorizontalAlignment = -4108
        $summarySheet.Rows.Item(2).RowHeight = 24

        $summaryHeaders = @('통합 분류', '학습용 라벨', '묶는 기준', '공시 유형 수', '문서 수', '문서 비중')
        for ($column = 1; $column -le 6; $column++) {
            $summarySheet.Cells.Item(4, $column).Value2 = $summaryHeaders[$column - 1]
        }
        $summarySheet.Range('A4:F4').Interior.Color = Get-OleColor 68 114 196
        $summarySheet.Range('A4:F4').Font.Color = Get-OleColor 255 255 255
        $summarySheet.Range('A4:F4').Font.Bold = $true
        $summarySheet.Range('A4:F4').HorizontalAlignment = -4108
        $summarySheet.Rows.Item(4).RowHeight = 26

        $groupRows = @(
            @('자본·주식관리', 'EQUITY_MANAGEMENT', '자기주식·자사주신탁·감자·무상증자 등 회사 자체의 주식·자본구조 조정'),
            @('자금조달·메자닌', 'FINANCING_SECURITIES', '유상증자와 CB·EB·조건부자본증권의 발행·매도·권리행사 등 조달증권 생애주기'),
            @('기업결합·조직재편', 'CORPORATE_RESTRUCTURING', '회사·사업의 합병·분할·분할합병·주식교환·영업양수 등 조직 재편'),
            @('투자·자산거래', 'INVESTMENT_ASSET_TRANSACTIONS', '타법인 지분이나 유형자산의 취득·매각'),
            @('상장·시장변경', 'LISTING_STATUS', '해외 증권시장의 상장 또는 상장폐지 결정·완료'),
            @('경영위험', 'BUSINESS_RISK', '소송 제기나 영업정지처럼 법률·영업연속성 위험이 현실화된 사건')
        )

        for ($index = 0; $index -lt $groupRows.Count; $index++) {
            $row = $index + 5
            $summarySheet.Cells.Item($row, 1).Value2 = $groupRows[$index][0]
            $summarySheet.Cells.Item($row, 2).Value2 = $groupRows[$index][1]
            $summarySheet.Cells.Item($row, 3).Value2 = $groupRows[$index][2]
            $summarySheet.Cells.Item($row, 4).Formula = "=COUNTIF('AI 학습용 분류'!`$C`$2:`$C`$29,A$row)"
            $summarySheet.Cells.Item($row, 5).Formula = "=SUMIF('AI 학습용 분류'!`$C`$2:`$C`$29,A$row,'AI 학습용 분류'!`$B`$2:`$B`$29)"
            $summarySheet.Cells.Item($row, 6).Formula = "=E$row/SUM(`$E`$5:`$E`$10)"
            $fillColor = $groupColors[$groupRows[$index][0]]
            $summarySheet.Range("A$row:B$row").Interior.Color = $fillColor
            $summarySheet.Cells.Item($row, 1).Font.Bold = $true
        }

        $summarySheet.Cells.Item(11, 1).Value2 = '합계'
        $summarySheet.Cells.Item(11, 4).Formula = '=SUM(D5:D10)'
        $summarySheet.Cells.Item(11, 5).Formula = '=SUM(E5:E10)'
        $summarySheet.Cells.Item(11, 6).Formula = '=SUM(F5:F10)'
        $summarySheet.Range('A11:F11').Font.Bold = $true
        $summarySheet.Range('A11:F11').Interior.Color = Get-OleColor 217 225 242
        $summarySheet.Range('A11:F11').Borders.Item(8).LineStyle = 1
        $summarySheet.Range('A11:F11').Borders.Item(8).Weight = 2

        $summarySheet.Range('A13:F13').Merge()
        $summarySheet.Range('A13').Value2 = "사용 방법: 'AI 학습용 분류' 시트의 E열(학습용 라벨) 6개만 모델의 정답값(target)으로 사용합니다."
        $summarySheet.Range('A14:F14').Merge()
        $summarySheet.Range('A14').Value2 = "주의: 상위 두 그룹이 전체의 85% 이상이므로 학습·평가 분할 시 stratified split과 class weight를 권장합니다."
        $summarySheet.Range('A15:F15').Merge()
        $summarySheet.Range('A15').Value2 = '분류 원칙: 일반 단어보다 긴 핵심 구문(자기전환사채, 타법인주식, 주식교환·이전, 상장폐지)을 먼저 판정합니다.'
        $summarySheet.Range('A13:F15').Interior.Color = Get-OleColor 242 242 242
        $summarySheet.Range('A13:F15').Font.Color = Get-OleColor 89 89 89
        $summarySheet.Range('A13:F15').WrapText = $true
        $summarySheet.Range('A13:F15').HorizontalAlignment = -4131

        $summarySheet.Range('A4:F11').VerticalAlignment = -4108
        $summarySheet.Range('A5:A11').HorizontalAlignment = -4131
        $summarySheet.Range('B5:B10').HorizontalAlignment = -4108
        $summarySheet.Range('C5:C10').HorizontalAlignment = -4131
        $summarySheet.Range('D5:F11').HorizontalAlignment = -4152
        $summarySheet.Range('D5:E11').NumberFormat = '#,##0'
        $summarySheet.Range('F5:F11').NumberFormat = '0.0%'
        $summarySheet.Range('A4:F11').Borders.Color = Get-OleColor 217 217 217
        $summarySheet.Range('A4:F11').Borders.LineStyle = 1
        $summarySheet.Columns.Item('A').ColumnWidth = 22
        $summarySheet.Columns.Item('B').ColumnWidth = 35
        $summarySheet.Columns.Item('C').ColumnWidth = 62
        $summarySheet.Columns.Item('D').ColumnWidth = 13
        $summarySheet.Columns.Item('E').ColumnWidth = 12
        $summarySheet.Columns.Item('F').ColumnWidth = 12
        $summarySheet.Range('5:10').RowHeight = 38
        $summarySheet.Range('11:11').RowHeight = 24
        $summarySheet.Range('13:15').RowHeight = 26
        $summarySheet.Tab.Color = Get-OleColor 68 114 196
        Set-WindowStyle -Excel $excel -Worksheet $summarySheet -SplitRow 4

        $summarySheet.PageSetup.Orientation = 2
        $summarySheet.PageSetup.Zoom = $false
        $summarySheet.PageSetup.FitToPagesWide = 1
        $summarySheet.PageSetup.FitToPagesTall = 1
        $summarySheet.PageSetup.PrintArea = '$A$1:$F$15'
        $summarySheet.PageSetup.LeftMargin = $excel.InchesToPoints(0.25)
        $summarySheet.PageSetup.RightMargin = $excel.InchesToPoints(0.25)
        $summarySheet.PageSetup.TopMargin = $excel.InchesToPoints(0.4)
        $summarySheet.PageSetup.BottomMargin = $excel.InchesToPoints(0.4)
        Write-Output 'STEP 7 summary_sheet_styled'

        Write-Output 'STEP 8 before_calculation'
        $excel.CalculateFullRebuild()
        Write-Output 'STEP 9 calculation_done'
        $dataSheet.Activate()
        $workbook.SaveAs($outputPath, 51)
        Write-Output 'STEP 10 workbook_saved'

        Export-WorksheetPng -Worksheet $dataSheet -Path $dataPreviewPath
        Write-Output 'STEP 11 data_preview_saved'
        Export-WorksheetPng -Worksheet $summarySheet -Path $summaryPreviewPath
        Write-Output 'STEP 12 summary_preview_saved'

        [pscustomobject]@{
            mode = $Mode
            output = $outputPath
            source_types = 28
            total_documents = [int]$excel.WorksheetFunction.Sum($dataSheet.Range('B2:B29'))
            previews = @($dataPreviewPath, $summaryPreviewPath)
        } | ConvertTo-Json -Compress
    }
    finally {
        if ($null -ne $workbook) { try { $workbook.Close($true) } catch { } }
        if ($null -ne $sourceWorkbook) { try { $sourceWorkbook.Close($false) } catch { } }
        if ($null -ne $excel) { try { $excel.Quit() } catch { } }
        Release-ComObject $dataTable
        Release-ComObject $summarySheet
        Release-ComObject $dataSheet
        Release-ComObject $workbook
        Release-ComObject $sourceRange
        Release-ComObject $sourceWorksheet
        Release-ComObject $sourceWorkbook
        Release-ComObject $excel
        [GC]::Collect()
        [GC]::WaitForPendingFinalizers()
    }
    exit 0
}

if ($Mode -eq 'VerifyFinal') {
    $excel = $null
    $workbook = $null
    $dataSheet = $null
    $summarySheet = $null
    try {
        if (-not (Test-Path -LiteralPath $outputPath)) { throw "Output workbook not found: $outputPath" }
        $excel = New-Object -ComObject Excel.Application
        $excel.Visible = $false
        $excel.DisplayAlerts = $false
        $excel.ScreenUpdating = $true
        $workbook = $excel.Workbooks.Open($outputPath)
        $dataSheet = $workbook.Worksheets.Item('AI 학습용 분류')
        $summarySheet = $workbook.Worksheets.Item('그룹 기준')
        $excel.CalculateFullRebuild()

        $rowCount = [int]$dataSheet.UsedRange.Rows.Count
        $columnCount = [int]$dataSheet.UsedRange.Columns.Count
        $totalDocuments = [int]$excel.WorksheetFunction.Sum($dataSheet.Range('B2:B29'))
        $labels = [System.Collections.Generic.HashSet[string]]::new()
        $formulaErrors = [System.Collections.Generic.List[string]]::new()
        for ($row = 2; $row -le 29; $row++) {
            [void]$labels.Add([string]$dataSheet.Cells.Item($row, 5).Value2)
        }
        foreach ($sheet in @($dataSheet, $summarySheet)) {
            $usedRange = $sheet.UsedRange
            for ($row = 1; $row -le $usedRange.Rows.Count; $row++) {
                for ($column = 1; $column -le $usedRange.Columns.Count; $column++) {
                    $text = [string]$usedRange.Cells.Item($row, $column).Text
                    if ($text -match '^#(REF!|DIV/0!|VALUE!|NAME\?|N/A|NUM!)$') {
                        $formulaErrors.Add("$($sheet.Name)!$($usedRange.Cells.Item($row, $column).Address()):$text")
                    }
                }
            }
            Release-ComObject $usedRange
        }

        $summaryTypes = [int]$summarySheet.Range('D11').Value2
        $summaryDocuments = [int]$summarySheet.Range('E11').Value2
        $summaryShare = [double]$summarySheet.Range('F11').Value2
        if ($rowCount -ne 29 -or $columnCount -ne 5) { throw "Unexpected data dimensions: ${rowCount}x${columnCount}" }
        if ($totalDocuments -ne 598) { throw "Unexpected document total: $totalDocuments" }
        if ($labels.Count -ne 6) { throw "Unexpected label count: $($labels.Count)" }
        if ($summaryTypes -ne 28 -or $summaryDocuments -ne 598 -or [Math]::Abs($summaryShare - 1) -gt 0.000001) {
            throw "Summary totals do not reconcile."
        }
        if ($formulaErrors.Count -gt 0) { throw "Formula errors found: $($formulaErrors -join ', ')" }

        Export-WorksheetPng -Worksheet $dataSheet -Path $dataPreviewPath
        Export-WorksheetPng -Worksheet $summarySheet -Path $summaryPreviewPath
        $dataSheet.Activate()
        $workbook.Save()

        [pscustomobject]@{
            mode = $Mode
            output = $outputPath
            sheets = @($dataSheet.Name, $summarySheet.Name)
            data_rows = $rowCount - 1
            total_documents = $totalDocuments
            label_count = $labels.Count
            summary_types = $summaryTypes
            summary_documents = $summaryDocuments
            formula_errors = $formulaErrors.Count
        } | ConvertTo-Json -Compress
    }
    finally {
        if ($null -ne $workbook) { try { $workbook.Close($true) } catch { } }
        if ($null -ne $excel) { try { $excel.Quit() } catch { } }
        Release-ComObject $summarySheet
        Release-ComObject $dataSheet
        Release-ComObject $workbook
        Release-ComObject $excel
        [GC]::Collect()
        [GC]::WaitForPendingFinalizers()
    }
}
