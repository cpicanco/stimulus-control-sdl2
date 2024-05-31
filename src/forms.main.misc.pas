unit forms.main.misc;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, StdCtrls,
  IniPropStorage, Spin, ComCtrls, ExtCtrls;

type

  { TFormMisc }

  TFormMisc = class(TForm)
    ButtonCreateRelease: TButton;
    ButtonRestoreConfigurationsBackup: TButton;
    ButtonDoConfigurationsBackup: TButton;
    ButtonTestDispenser: TButton;
    CheckBoxHideMouse: TCheckBox;
    CheckBoxShowMarkers: TCheckBox;
    CheckBoxShowModalFormForSpeechResponses: TCheckBox;
    CheckBoxTestMode: TCheckBox;
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
    LabelTestMode: TLabel;
    LabelEyeTracker: TLabel;
    LabelInterTrialInterval: TLabel;
    LabelLimitedHold: TLabel;
    LabelTimeOut: TLabel;
    PageControl: TPageControl;
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
    procedure FormCreate(Sender: TObject);
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
    0: {do nothing};
    1 : RS232.Dispenser(disp1);
    2 : RS232.Dispenser(disp2);
    3 : RS232.Dispenser(disp3);
    4 : RS232.Dispenser(disp4);
  end;
  RS232.Free;
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

