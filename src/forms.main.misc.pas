{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit forms.main.misc;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, StdCtrls,
  IniPropStorage, Spin, ComCtrls, ExtCtrls, Dialogs;

type

  { TFormMisc }

  TFormMisc = class(TForm)
    ButtonCreateRelease: TButton;
    ButtonRestoreConfigurationsBackup: TButton;
    ButtonDoConfigurationsBackup: TButton;
    ButtonTestDispenser: TButton;
    CheckBoxForceLastMonitor: TCheckBox;
    CheckBoxUseRemoteServer: TCheckBox;
    CheckBoxUseGazeAsInput: TCheckBox;
    CheckBoxHideMouse: TCheckBox;
    CheckBoxShowMarkers: TCheckBox;
    CheckBoxShowModalFormForSpeechResponses: TCheckBox;
    CheckBoxSimultaneousMTS: TCheckBox;
    CheckBoxTestMode: TCheckBox;
    ColorDialog: TColorDialog;
    ComboBoxAudioFolder: TComboBox;
    ComboBoxAudioPromptForText: TComboBox;
    ComboBoxController: TComboBox;
    ComboBoxFixedSamplePosition: TComboBox;
    ComboBoxFontName: TComboBox;
    ComboBoxShouldRestartAt: TComboBox;
    ComboBoxEyeTracker: TComboBox;
    ComboBoxMonitor: TComboBox;
    FloatSpinEditScreenWidth: TFloatSpinEdit;
    FloatSpinEditCellsSize: TFloatSpinEdit;
    IniPropStorage1: TIniPropStorage;
    Label1: TLabel;
    LabelForceLastMonitor: TLabel;
    LabelUseRemoteServer: TLabel;
    LabelUseGazeAsInput: TLabel;
    LabelFontColor: TLabel;
    LabelScreenWidth: TLabel;
    LabelAudioFolder: TLabel;
    LabelAudioLoopInterval: TLabel;
    LabelAudioPromptForText: TLabel;
    LabelDefaultAudioLoops: TLabel;
    LabelFixedSamplePosition: TLabel;
    LabelFont: TLabel;
    LabelFontsize: TLabel;
    LabelHideMouse: TLabel;
    LabelAprilTagsSize: TLabel;
    LabelController: TLabel;
    LabelRecordingSeconds: TLabel;
    LabelCellsSize: TLabel;
    LabelShoudRestartAtBlockStart: TLabel;
    LabelMonitor: TLabel;
    LabelShowModal: TLabel;
    LabelSimultaneousMTS: TLabel;
    LabelTestMode: TLabel;
    LabelEyeTracker: TLabel;
    LabelInterTrialInterval: TLabel;
    LabelLimitedHold: TLabel;
    LabelTimeOut: TLabel;
    PageControl: TPageControl;
    PanelFontColor: TPanel;
    RadioGroupDispenser: TRadioGroup;
    SpinEditAprilTagsSize: TSpinEdit;
    SpinEditAudioLoopInterval: TSpinEdit;
    SpinEditDefaultAudioLoops: TSpinEdit;
    SpinEditFontSize: TSpinEdit;
    SpinEditInterTrialInterval: TSpinEdit;
    SpinEditLimitedHold: TSpinEdit;
    SpinEditRecordingSeconds: TSpinEdit;
    SpinEditTimeOut: TSpinEdit;
    TabSheetOther: TTabSheet;
    TabSheetPseudowords: TTabSheet;
    TabSheetControllers: TTabSheet;
    TabSheetGeneral: TTabSheet;
    TabSheetEyeTracking: TTabSheet;
    procedure ButtonCreateReleaseClick(Sender: TObject);
    procedure ButtonDoConfigurationsBackupClick(Sender: TObject);
    procedure ButtonRestoreConfigurationsBackupClick(Sender: TObject);
    procedure ButtonTestDispenserClick(Sender: TObject);
    procedure CheckBoxSimultaneousMTSChange(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure PanelFontColorClick(Sender: TObject);
  private

  public
    procedure Initialize;
    procedure Finalize;
  end;

var
  FormMisc: TFormMisc;

implementation

uses sdl.app, session.fileutils, Devices.RS232i, ui.backup, release;

{$R *.lfm}

{ TFormMisc }

procedure TFormMisc.FormCreate(Sender: TObject);
var
  LStringList : TStringList;
  i: Integer;
begin
  GetAudioFilesFor(ComboBoxAudioPromptForText.Items);
  GetAudioFoldersFor(ComboBoxAudioFolder.Items);
  GetFontFilesFor(ComboBoxFontName.Items);

  LStringList := TStringList.Create;
  try
    TSDLApplication.GetAvailableMonitors(LStringList);
    if ComboBoxMonitor.Items.Count = LStringList.Count then begin
      for i := 0 to ComboBoxMonitor.Items.Count -1 do begin
        if ComboBoxMonitor.Items[i] <> LStringList[i] then begin
           ComboBoxMonitor.Items.Clear;
           ComboBoxMonitor.Items.Assign(LStringList);
        end;
      end;
    end else begin
      ComboBoxMonitor.Items.Assign(LStringList);
    end;
  finally
    LStringList.Clear;
    LStringList.Free;
  end;
end;

procedure TFormMisc.PanelFontColorClick(Sender: TObject);
begin
  ColorDialog.Title := 'Escolha a cor do fundo';
  if ColorDialog.Execute then begin
    with PanelFontColor do begin
      Color := ColorDialog.Color;
    end;
  end;

  ColorDialog.Title := 'Escolha a cor da fonte';
  if ColorDialog.Execute then begin
    with PanelFontColor do begin
      Font.Color := ColorDialog.Color;
    end;
  end;
end;

procedure TFormMisc.Initialize;
begin
  case RadioGroupDispenser.ItemIndex of
    0: { do nothing };
    otherwise begin
      RS232 := TRS232.Create;
      case RadioGroupDispenser.ItemIndex of
        1 : RS232.DefaultDispenser := disp1;
        2 : RS232.DefaultDispenser := disp2;
        3 : RS232.DefaultDispenser := disp3;
        4 : RS232.DefaultDispenser := disp4;
      end;
    end;
  end;
end;

procedure TFormMisc.Finalize;
begin
  case RadioGroupDispenser.ItemIndex of
    0: { do nothing };
    otherwise begin
      RS232.Free;
      RS232 := nil;
    end;
  end;
end;

procedure TFormMisc.ButtonTestDispenserClick(Sender: TObject);
begin
  RS232 := TRS232.Create;
  case RadioGroupDispenser.ItemIndex of
    0 : { do nothing };
    1 : RS232.Dispenser(disp1);
    2 : RS232.Dispenser(disp2);
    3 : RS232.Dispenser(disp3);
    4 : RS232.Dispenser(disp4);
  end;
  RS232.Free;
end;

procedure TFormMisc.CheckBoxSimultaneousMTSChange(Sender: TObject);
begin
  with CheckBoxSimultaneousMTS do begin
    if Checked then begin
      Caption := 'Simultâneo';
    end else begin
      Caption := 'Com atraso 0';
    end;
  end;
end;

procedure TFormMisc.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  IniPropStorage1.Save;
end;

procedure TFormMisc.ButtonDoConfigurationsBackupClick(Sender: TObject);
begin
  DoConfigurationsBackup;
end;

procedure TFormMisc.ButtonCreateReleaseClick(Sender: TObject);
begin
  CreateReleaseFolder;
end;

procedure TFormMisc.ButtonRestoreConfigurationsBackupClick(Sender: TObject);
begin
  RestoreConfigurationsBackup;
end;

{ TFormMisc }

end.

