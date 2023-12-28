unit forms.main.misc;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, StdCtrls,
  IniPropStorage, Spin, ComCtrls;

type

  { TFormMisc }

  TFormMisc = class(TForm)
    CheckBoxShowMarkers: TCheckBox;
    CheckBoxShowModalFormForSpeechResponses: TCheckBox;
    CheckBoxTestMode: TCheckBox;
    ComboBoxController: TComboBox;
    ComboBoxShouldRestartAt: TComboBox;
    ComboBoxEyeTracker: TComboBox;
    ComboBoxAudioFolder: TComboBox;
    ComboBoxAudioPromptForText: TComboBox;
    ComboBoxFixedSamplePosition: TComboBox;
    ComboBoxFontName: TComboBox;
    ComboBoxMonitor: TComboBox;
    IniPropStorage1: TIniPropStorage;
    Label1: TLabel;
    LabelController: TLabel;
    LabelShoudRestartAtBlockStart: TLabel;
    LabelMonitor: TLabel;
    LabelTestMode: TLabel;
    LabelEyeTracker: TLabel;
    LabelAudioFolder: TLabel;
    LabelAudioPromptForText: TLabel;
    LabelFixedSamplePosition: TLabel;
    LabelFont: TLabel;
    LabelFontsize: TLabel;
    LabelInterTrialInterval: TLabel;
    LabelLimitedHold: TLabel;
    LabelRecordingSeconds: TLabel;
    LabelShowModal: TLabel;
    LabelTimeOut: TLabel;
    PageControl: TPageControl;
    SpinEditFontSize: TSpinEdit;
    SpinEditInterTrialInterval: TSpinEdit;
    SpinEditLimitedHold: TSpinEdit;
    SpinEditRecordingSeconds: TSpinEdit;
    SpinEditTimeOut: TSpinEdit;
    TabSheetControllers: TTabSheet;
    TabSheetGeneral: TTabSheet;
    TabSheetEyeTracking: TTabSheet;
    procedure ComboBoxDesignFolderChange(Sender: TObject);
    procedure FormCreate(Sender: TObject);
  private

  public

  end;

var
  FormMisc: TFormMisc;

implementation

uses sdl.app, session.pool, session.fileutils;

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

procedure TFormMisc.ComboBoxDesignFolderChange(Sender: TObject);
begin

end;

{ TFormMisc }

end.

