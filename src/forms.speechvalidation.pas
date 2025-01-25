{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit forms.speechvalidation;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, Grids, StdCtrls,
  session.loggers.types;

type

  { TFormSpeechValidationQueue }

  TFormSpeechValidationQueue = class(TForm)
    EditSpeech: TEdit;
    StringGridSpeechEvents: TStringGrid;
    procedure EditSpeechKeyPress(Sender: TObject; var Key: char);
  private
    FExpectedText: TTimestampedEvent;
    procedure SetExpectedText(AValue: TTimestampedEvent);
  public
    property ExpectedText : TTimestampedEvent read FExpectedText write SetExpectedText;
  end;

var
  FormSpeechValidationQueue: TFormSpeechValidationQueue;

implementation

uses session.loggers.writerow.timestamp, timestamps, timestamps.methods;

{$R *.lfm}

{ TFormSpeechValidationQueue }

procedure TFormSpeechValidationQueue.EditSpeechKeyPress(Sender: TObject;
  var Key: char);
var
  LRow : integer;
begin
  case Ord(Key) of
    13 : begin
      with StringGridSpeechEvents do begin
        if RowCount = 1 then Exit;
        LRow := RowCount -1;
        if EditSpeech.Text = Cells[4, LRow] then begin
          Cells[3, LRow] := 'Correct.Response';
        end else begin
          Cells[3, LRow] := 'Incorrect.Response';
          Cells[4, LRow] := Cells[4, LRow] + '-' + EditSpeech.Text;
        end;

        Timestamp(
          Cells[0, LRow],
          Cells[1, LRow],
          Cells[2, LRow],
          Cells[3, LRow],
          Cells[4, LRow]);
        RowCount := RowCount-1;
      end;
    end;
  end;
end;

procedure TFormSpeechValidationQueue.SetExpectedText(AValue: TTimestampedEvent);
begin
  if FExpectedText.Timestamp = AValue.Timestamp then Exit;
  FExpectedText := AValue;

  with StringGridSpeechEvents do begin
    InsertRowWithValues(1,[
      AValue.Timestamp.ToString,
      AValue.Trial.ToString,
      AValue.Block.ToString,
      AValue.Code,
      AValue.Annotation]);
  end;
end;

end.

